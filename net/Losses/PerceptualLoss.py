import torch
import torch.nn.functional as F

# --- Perceptual loss network  --- #
# 使用的一致性感知损失函数
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        #转化为使用vgg_model进行损失函数的测试等--
        self.vgg_layers = vgg_model
        #vgg layer的映射
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #返回vgg中可以用作感知损失的几层layer
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        # 对他们的feature进行损失 使用vgg架构得到两张不同图片的特征
        # 对图片的特征进行L1损失值的计算求解
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)