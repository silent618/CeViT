from __future__ import print_function, division

from numpy import *

import torch
import torch.nn as nn

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, ViTForImageClassification
from peft import LoraConfig, LoraModel

from utils.common_utils import print_trainable_parameters
import torch.nn.functional as F
from torch.distributions import normal
from torchvision import models


class my_normal(nn.Module):
    def __init__(self, mu, sigma):
        super(my_normal, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.normal_distribution = normal.Normal(self.mu, self.sigma)

    def forward(self, x):
        return self.normal_distribution.cdf(x)


class AdapterLayer(nn.Module):
    def __init__(self, in_features=768, reduction_factor=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_factor),
            nn.ReLU(),
            nn.Linear(in_features // reduction_factor, in_features)
        )

    def forward(self, x):
        return self.adapter(x)


class ViTOutput(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class CoViT_Adapter(nn.Module):
    def __init__(self, device='cuda:0', rank_LoRa=8):
        super(CoViT_Adapter, self).__init__()

        self.device = device

        self.model_LoRa = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
        )
        self.model_LoRa.classifier = nn.Identity()

        config = LoraConfig(
            r=rank_LoRa,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model_LoRa = LoraModel(self.model_LoRa, config, "LoRa")
        print("parameter count for model_LoRa: ")
        print_trainable_parameters(self.model_LoRa)

        # 创建 Adapter 模块列表
        # self.adapters = nn.ModuleList([AdapterLayer() for i in range(nb_tasks)])
        self.adapters = nn.ModuleList([nn.ModuleList([AdapterLayer() for i in range(2)]) for j in range(12)])

        self.fc_classification = nn.Linear(768, 1)
        self.fc_regression = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_vec = torch.tensor([]).to(self.device)

        x_left = x[0]
        x_right = x[1]

        hidden_states_left = self.model_LoRa.model.vit.embeddings(x_left)

        for j, block in enumerate(self.model_LoRa.model.vit.encoder.layer):
            # print("j: ",j)

            hidden_states_ln_left = block.layernorm_before(hidden_states_left)
            self_attention_outputs_left = block.attention(hidden_states_ln_left, head_mask=None, output_attentions=False)

            attention_output_left = self_attention_outputs_left[0]
            outputs_left = self_attention_outputs_left[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_left = attention_output_left + hidden_states_left

            # Adapter after the self-attention output
            adapted_hidden_states_left = self.adapters[j][0](hidden_states_left)

            # in ViT, layernorm is also applied after self-attention
            layer_output_left = block.layernorm_after(hidden_states_left)

            # MLP block
            layer_output_left = block.intermediate(layer_output_left)

            # second residual connection is done here
            layer_output_left = block.output(layer_output_left, hidden_states_left)

            layer_output_left = 0.1 * adapted_hidden_states_left + layer_output_left

            # Residual connection
            hidden_states_left = (layer_output_left,) + outputs_left

            hidden_states_left = hidden_states_left[0]

        # Final LayerNorm
        hidden_states_left = self.model_LoRa.model.vit.layernorm(hidden_states_left)

        # pooled_output = hidden_states[:, 0]
        # x_i = self.relu(self.model_LoRa.classifier(pooled_output))

        logits_left = self.model_LoRa.classifier(hidden_states_left[:, 0, :])

        # Calculate binary classification output
        out_left = self.fc_classification(logits_left)
        y_classification_left = self.sigmoid(out_left)

        # Calculate regression output
        y_regression_left = self.fc_regression(logits_left)

        hidden_states_right = self.model_LoRa.model.vit.embeddings(x_right)
        for j, block in enumerate(self.model_LoRa.model.vit.encoder.layer):

            hidden_states_ln_right = block.layernorm_before(hidden_states_right)
            self_attention_outputs_right = block.attention(hidden_states_ln_right, head_mask=None,
                                                          output_attentions=False)

            attention_output_right = self_attention_outputs_right[0]
            outputs_right = self_attention_outputs_right[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_right = attention_output_right + hidden_states_right

            # Adapter after the self-attention output
            adapted_hidden_states_right = self.adapters[j][1](hidden_states_right)

            # in ViT, layernorm is also applied after self-attention
            layer_output_right = block.layernorm_after(hidden_states_right)

            # MLP block
            layer_output_right = block.intermediate(layer_output_right)

            # second residual connection is done here
            layer_output_right = block.output(layer_output_right, hidden_states_right)

            layer_output_right = 0.1 * adapted_hidden_states_right + layer_output_right

            # Residual connection
            hidden_states_right = (layer_output_right,) + outputs_right

            hidden_states_right = hidden_states_right[0]

        # Final LayerNorm
        hidden_states_right = self.model_LoRa.model.vit.layernorm(hidden_states_right)

        # pooled_output = hidden_states[:, 0]
        # x_i = self.relu(self.model_LoRa.classifier(pooled_output))

        logits_right = self.model_LoRa.classifier(hidden_states_right[:, 0, :])

        # Calculate binary classification output
        out_right = self.fc_classification(logits_right)
        y_classification_right = self.sigmoid(out_right)

        # Calculate regression output
        y_regression_right = self.fc_regression(logits_right)

        y_vec = torch.cat((y_vec, y_classification_left, y_regression_left, y_classification_right, y_regression_right), -1)
        return y_vec[:, 0:2], y_vec[:, 2:4]


class MultiTaskMLP(nn.Module):
    def __init__(self, input_size=3 * 112 * 112, hidden_sizes=[4096, 2048, 1024],
                 dropout_prob=0.5, device='cuda:0'):
        super(MultiTaskMLP, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义三层MLP
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        # 分类头
        self.classifier = nn.Linear(hidden_sizes[2], 1)

        # 回归头
        self.regressor = nn.Linear(hidden_sizes[2], 1)

        # Dropout层
        self.dropout = nn.Dropout(dropout_prob)

        self.normal_cdf = my_normal(0, 1)

        self.bn_classifier = nn.BatchNorm1d(hidden_sizes[2])
        self.bn_regressor = nn.BatchNorm1d(hidden_sizes[2])

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]

        x_left = self.max_pool(x_left)
        # 展平输入图像
        x_left = x_left.view(x_left.size(0), -1)  # (batch_size, 3*224*224)

        # 第一层全连接 + ReLU + Dropout
        x_left = F.relu(self.fc1(x_left))
        x_left = self.dropout(x_left)

        # 第二层全连接 + ReLU + Dropout
        x_left = F.relu(self.fc2(x_left))
        x_left = self.dropout(x_left)

        # 第三层全连接 + ReLU
        x_left = F.relu(self.fc3(x_left))

        y_classification_left = self.normal_cdf(self.classifier(self.bn_classifier(x_left)))
        y_regression_left = self.regressor(self.bn_regressor(x_left))

        x_right = self.max_pool(x_right)
        # 展平输入图像
        x_right = x_right.view(x_right.size(0), -1)  # (batch_size, 3*224*224)

        # 第一层全连接 + ReLU + Dropout
        x_right = F.relu(self.fc1(x_right))
        x_right = self.dropout(x_right)

        # 第二层全连接 + ReLU + Dropout
        x_right = F.relu(self.fc2(x_right))
        x_right = self.dropout(x_right)

        # 第三层全连接 + ReLU
        x_right = F.relu(self.fc3(x_right))

        y_classification_right = self.normal_cdf(self.classifier(self.bn_classifier(x_right)))
        y_regression_right = self.regressor(self.bn_regressor(x_right))

        y_vec = torch.cat((y_classification_left, y_regression_left, y_classification_right, y_regression_right), -1)

        return y_vec[:, 0:2], y_vec[:, 2:]


class CoViT(nn.Module):
    def __init__(self, device='cuda:0', rank_LoRa=8):
        super(CoViT, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
        )

        self.model.classifier = nn.Identity()

        config = LoraConfig(
            r=rank_LoRa,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model = LoraModel(self.model, config, "LoRa")
        print("parameter count for model: ")
        print_trainable_parameters(self.model)

        self.fc_classification = nn.Linear(768, 1)
        self.fc_regression = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]

        hidden_states_left = self.model.model.vit.embeddings(x_left)
        hidden_states_right = self.model.model.vit.embeddings(x_right)

        for j, block in enumerate(self.model.model.vit.encoder.layer):
            """处理左眼图像"""
            hidden_states_ln_left = block.layernorm_before(hidden_states_left)
            self_attention_outputs_left = block.attention(hidden_states_ln_left, head_mask=None,
                                                          output_attentions=False)

            attention_output_left = self_attention_outputs_left[0]
            outputs_left = self_attention_outputs_left[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_left = attention_output_left + hidden_states_left

            # in ViT, layernorm is also applied after self-attention
            layer_output_left = block.layernorm_after(hidden_states_left)

            layer_output_left = block.intermediate(layer_output_left)

            # second residual connection is done here
            layer_output_left = block.output(layer_output_left, hidden_states_left)

            # Residual connection
            hidden_states_left = (layer_output_left,) + outputs_left

            hidden_states_left = hidden_states_left[0]
            """处理右眼图像"""
            hidden_states_ln_right = block.layernorm_before(hidden_states_right)
            self_attention_outputs_right = block.attention(hidden_states_ln_right, head_mask=None,
                                                           output_attentions=False)

            attention_output_right = self_attention_outputs_right[0]
            outputs_right = self_attention_outputs_right[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_right = attention_output_right + hidden_states_right

            # in ViT, layernorm is also applied after self-attention
            layer_output_right = block.layernorm_after(hidden_states_right)

            layer_output_right = block.intermediate(layer_output_right)

            # second residual connection is done here
            layer_output_right = block.output(layer_output_right, hidden_states_right)

            # Residual connection
            hidden_states_right = (layer_output_right,) + outputs_right

            hidden_states_right = hidden_states_right[0]

        # Final LayerNorm
        hidden_states_left = self.model.model.vit.layernorm(hidden_states_left)
        hidden_states_right = self.model.model.vit.layernorm(hidden_states_right)

        logits_left = self.model.model.classifier(hidden_states_left[:, 0, :])
        # Calculate binary classification output
        out_left = self.fc_classification(logits_left)
        y_classification_left = self.sigmoid(out_left)
        # Calculate regression output
        y_regression_left = self.fc_regression(logits_left)

        logits_right = self.model.model.classifier(hidden_states_right[:, 0, :])
        # Calculate binary classification output
        out_right = self.fc_classification(logits_right)
        y_classification_right = self.sigmoid(out_right)
        # Calculate regression output
        y_regression_right = self.fc_regression(logits_right)

        y_vec = torch.cat((y_classification_left, y_regression_left, y_classification_right, y_regression_right), -1)

        return y_vec[:, 0:2], y_vec[:, 2:]


class CoViT_Normal(nn.Module):
    def __init__(self, device='cuda:0', rank_LoRa=8):
        super(CoViT_Normal, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
        )

        self.model.classifier = nn.Identity()

        config = LoraConfig(
            r=rank_LoRa,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model = LoraModel(self.model, config, "LoRa")
        print("parameter count for model: ")
        print_trainable_parameters(self.model)

        self.fc_classification = nn.Linear(768, 1)
        self.fc_regression = nn.Linear(768, 1)
        self.sigmoid = my_normal(0, 1)

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]

        hidden_states_left = self.model.model.vit.embeddings(x_left)
        hidden_states_right = self.model.model.vit.embeddings(x_right)

        for j, block in enumerate(self.model.model.vit.encoder.layer):
            """处理左眼图像"""
            hidden_states_ln_left = block.layernorm_before(hidden_states_left)
            self_attention_outputs_left = block.attention(hidden_states_ln_left, head_mask=None,
                                                          output_attentions=False)

            attention_output_left = self_attention_outputs_left[0]
            outputs_left = self_attention_outputs_left[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_left = attention_output_left + hidden_states_left

            # in ViT, layernorm is also applied after self-attention
            layer_output_left = block.layernorm_after(hidden_states_left)

            layer_output_left = block.intermediate(layer_output_left)

            # second residual connection is done here
            layer_output_left = block.output(layer_output_left, hidden_states_left)

            # Residual connection
            hidden_states_left = (layer_output_left,) + outputs_left

            hidden_states_left = hidden_states_left[0]
            """处理右眼图像"""
            hidden_states_ln_right = block.layernorm_before(hidden_states_right)
            self_attention_outputs_right = block.attention(hidden_states_ln_right, head_mask=None,
                                                           output_attentions=False)

            attention_output_right = self_attention_outputs_right[0]
            outputs_right = self_attention_outputs_right[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_right = attention_output_right + hidden_states_right

            # in ViT, layernorm is also applied after self-attention
            layer_output_right = block.layernorm_after(hidden_states_right)

            layer_output_right = block.intermediate(layer_output_right)

            # second residual connection is done here
            layer_output_right = block.output(layer_output_right, hidden_states_right)

            # Residual connection
            hidden_states_right = (layer_output_right,) + outputs_right

            hidden_states_right = hidden_states_right[0]

        # Final LayerNorm
        hidden_states_left = self.model.model.vit.layernorm(hidden_states_left)
        hidden_states_right = self.model.model.vit.layernorm(hidden_states_right)

        logits_left = self.model.model.classifier(hidden_states_left[:, 0, :])
        # Calculate binary classification output
        out_left = self.fc_classification(logits_left)
        y_classification_left = self.sigmoid(out_left)
        # Calculate regression output
        y_regression_left = self.fc_regression(logits_left)

        logits_right = self.model.model.classifier(hidden_states_right[:, 0, :])
        # Calculate binary classification output
        out_right = self.fc_classification(logits_right)
        y_classification_right = self.sigmoid(out_right)
        # Calculate regression output
        y_regression_right = self.fc_regression(logits_right)

        y_vec = torch.cat((y_classification_left, y_regression_left, y_classification_right, y_regression_right), -1)

        return y_vec[:, 0:2], y_vec[:, 2:], out_left, out_right


if __name__ == '__main__':
    net = MultiTaskMLP()
    a = torch.randn(2, 3, 224, 224).to(torch.float)
    b = torch.randn(2, 3, 224, 224).to(torch.float)
    c = [a, b]
    out = net(c)
    d = 1
