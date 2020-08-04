import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, bias=True):
        super(Fusion, self).__init__()
        self.bias = bias

        # # -----------------------------  new fusion model 1 ----------------------------- # #
        self.img_con1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
        self.depth_con1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )


        # # -----------------------------  new fusion model 2 ----------------------------- # #
        self.img_con2= nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True),)
        self.depth_con2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, 1, padding=0), nn.ReLU(inplace=True), )



        # # -----------------------------  RFB 3 ----------------------------- # #
          # -----------------------------  GFU Layer 3 ----------------------- #

        self.cat_feature_out3 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(64, 1, 1, padding=0)
                                             )
        self.gate_rear3 = nn.Conv2d(512, 256, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.sconv3 = nn.Conv2d(256, 256, 1, padding=0)
        self.sconv_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.sconv_relu3 = nn.ReLU(inplace=True)

          # -----------------------------  RU 3  ----------------------------- #
        self.relu_RU3_img = nn.PReLU()
        self.conv_RU3_1_img = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_RU3_1_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU3_1_img = nn.PReLU()
        self.conv_RU3_2_img = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_RU3_2_img = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end3_img = nn.PReLU()

        self.relu_RU3_depth = nn.PReLU()
        self.conv_RU3_1_depth = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_RU3_1_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU3_1_depth = nn.PReLU()
        self.conv_RU3_2_depth = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_RU3_2_depth = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end3_depth = nn.PReLU()

        # # -----------------------------  RFB 4 ----------------------------- # #
          # -----------------------------  GFU Layer 2 ----------------------- #


        self.cat_feature_out4 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 1, 1, padding=0)
                                             )

        self.gate_rear4 = nn.Conv2d(256, 128, 1, padding=0)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.sconv4 = nn.Conv2d(128, 128, 1, padding=0)
        self.sconv_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.sconv_relu4 = nn.ReLU(inplace=True)

          # -----------------------------  RU 2  ----------------------------- #
        self.relu_RU4_img = nn.PReLU()
        self.conv_RU4_1_img = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_RU4_1_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU4_1_img = nn.PReLU()
        self.conv_RU4_2_img = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_RU4_2_img = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end4_img = nn.PReLU()

        self.relu_RU4_depth = nn.PReLU()
        self.conv_RU4_1_depth = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_RU4_1_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU4_1_depth = nn.PReLU()
        self.conv_RU4_2_depth = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_RU4_2_depth = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end4_depth = nn.PReLU()

        # # -----------------------------  RFB 5 ----------------------------- # #
        # -----------------------------  GFU Layer 1 ----------------------- #

        self.cat_feature_out5 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 1, 1, padding=0)
                                             )

        self.gate_rear5 = nn.Conv2d(128, 64, 1, padding=0)
        self.bn5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.sconv5 = nn.Conv2d(64, 64, 1, padding=0)
        self.sconv_bn5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.sconv_relu5 = nn.ReLU(inplace=True)

        # -----------------------------  RU 5  ----------------------------- #
        self.relu_RU5_img = nn.PReLU()
        self.conv_RU5_1_img = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_RU5_1_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU5_1_img = nn.PReLU()
        self.conv_RU5_2_img = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_RU5_2_img = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end5_img = nn.PReLU()

        self.relu_RU5_depth = nn.PReLU()
        self.conv_RU5_1_depth = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_RU5_1_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_RU5_1_depth = nn.PReLU()
        self.conv_RU5_2_depth = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_RU5_2_depth = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.relu_RU_end5_depth = nn.PReLU()

        # # -----------------------------  transition layer ----------------------------- # #

        self.tran5 = nn.Conv2d(1024, 512, 1, padding=0)
        self.tran4 = nn.Conv2d(1024, 256, 1, padding=0)
        self.tran3 = nn.Conv2d(512, 128, 1, padding=0)
        self.tran2 = nn.Conv2d(256, 64, 1, padding=0)

        # # -----------------------------  prediction ----------------------------- # #
        self.pred_ori5 = nn.Conv2d(1024, 64, 1, padding=0)
        self.pred_ori4 = nn.Conv2d(1024, 64, 1, padding=0)
        self.pred_ori3 = nn.Conv2d(512, 64, 1, padding=0)
        self.pred_ori2 = nn.Conv2d(256, 64, 1, padding=0)
        self.pred_ori1 = nn.Conv2d(128, 64, 1, padding=0)

        self.prediction5 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction4 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction3 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction2 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction1 = nn.Conv2d(64, 2, 1, padding=0)
        self.prediction = nn.Conv2d(64, 2, 1, padding=0)



        self.upsample4 = nn.Sequential(nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
                                       nn.ReLU(inplace=True),
                                      )
        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                       nn.ReLU(inplace=True),
                                      )
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                                       nn.ReLU(inplace=True),
                                      )
        self.upsample_img = nn.Sequential(nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                       nn.ReLU(inplace=True),
                                      )
        self.upsample_depth = nn.Sequential(nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                                       nn.ReLU(inplace=True),
                                      )


        ##--------------------------------module2-------------------------------##
        self.conv1_m = nn.Conv2d(64*5, 64, 3,padding=1)
        self.bn1_m = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_m = nn.RReLU()

        self.conv2_m = nn.Conv2d(64, 64, 3,padding=1)
        self.bn2_m = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_m = nn.RReLU()

        self.pre_m = nn.Conv2d(64, 5, 1,padding=0)
        self.vector_m = nn.AdaptiveAvgPool2d((1, 1))


        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self._initialize_weights()





    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, h1, h2, h3, h4, h5, d1, d2, d3, d4, d5):


      ##--------------------------conv module（交织模块）---------------------------------##
      img_feature5 = self.img_con1(h5)
      depth_feature5 = self.depth_con1(d5)
      feature5 = torch.cat((img_feature5,depth_feature5),1)


        ## Level Prediction
      prediction5 = self.pred_ori5(feature5)
      prediction5_L = prediction5
      prediction5 = self.prediction5(prediction5)
      prediction5 = F.upsample(prediction5, size=[256, 256], mode='bilinear')

      img_feature5 = self.upsample_img(img_feature5)
      depth_feature5 = self.upsample_depth(depth_feature5)


      img_feature4 = h4 + img_feature5 + depth_feature5
      depth_feature4 = d4 + depth_feature5 + img_feature5

      img_feature4 = self.img_con2(img_feature4)
      depth_feature4 = self.depth_con2(depth_feature4)
      feature4 = torch.cat((img_feature4,depth_feature4),1)

      ## out
      feature4 = self.upsample4(feature4)

        ## Level Prediction
      prediction4 = self.pred_ori4(feature4)
      prediction4_L = prediction4
      prediction4 = self.prediction4(prediction4)
      prediction4 = F.upsample(prediction4, size=[256, 256], mode='bilinear')

      feature4 = self.tran4(feature4)




      ##--------------------------Level 3 ---------------------------------##
      ## GFU
      cat_feature3 = torch.cat((h3, d3), 1)
      cat_feature3 = self.bn3(self.gate_rear3(cat_feature3))

      cat_feature3 = cat_feature3 + feature4

      cat_feature3 = self.relu3_3(cat_feature3)
      cat_feature3_in = self.sconv_relu3(self.sconv_bn3(self.sconv3(cat_feature3)))

      gate_out3 = torch.sigmoid(self.cat_feature_out3(cat_feature3_in))


      ## RU RGB
      RGB_feature = h3
      RGB_feature = self.relu_RU3_img(RGB_feature)
      RGB_feature = self.relu_RU3_1_img(self.bn_RU3_1_img(self.conv_RU3_1_img(RGB_feature)))
      RGB_feature = self.bn_RU3_2_img(self.conv_RU3_2_img(RGB_feature))
      RGB_feature3 = h3 + RGB_feature
      RGB_feature3 = self.relu_RU_end3_img(RGB_feature3)

      ## RU Depth
      Depth_feature = d3
      Depth_feature = self.relu_RU3_depth(Depth_feature)
      Depth_feature = self.relu_RU3_1_depth(self.bn_RU3_1_depth(self.conv_RU3_1_depth(Depth_feature)))
      Depth_feature = self.bn_RU3_2_depth(self.conv_RU3_2_depth(Depth_feature))
      Depth_feature3 = d3 + Depth_feature
      Depth_feature3 = self.relu_RU_end3_depth(Depth_feature3)

      ## out
      RGB_feature3 = torch.mul(gate_out3,RGB_feature3)
      Depth_feature3 = torch.mul(gate_out3, Depth_feature3)
      feature3 = torch.cat((RGB_feature3,Depth_feature3),1)
      feature3 = self.upsample3(feature3)

        ## Level Prediction
      prediction3 = self.pred_ori3(feature3)
      prediction3_L = prediction3
      prediction3 = self.prediction3(prediction3)
      prediction3 = F.upsample(prediction3, size=[256, 256], mode='bilinear')

      feature3 = self.tran3(feature3)






      ##--------------------------Level 4 ---------------------------------##
      ## GFU
      cat_feature2 = torch.cat((h2, d2), 1)

      cat_feature2 = self.bn4(self.gate_rear4(cat_feature2))

      cat_feature2 = cat_feature2 + feature3

      cat_feature2 = self.relu4_3(cat_feature2)
      cat_feature2_in = self.sconv_relu4(self.sconv_bn4(self.sconv4(cat_feature2)))

      gate_out2 = torch.sigmoid(self.cat_feature_out4(cat_feature2_in))


      ## RU RGB
      RGB_feature = h2
      RGB_feature = self.relu_RU4_img(RGB_feature)
      RGB_feature = self.relu_RU4_1_img(self.bn_RU4_1_img(self.conv_RU4_1_img(RGB_feature)))
      RGB_feature = self.bn_RU4_2_img(self.conv_RU4_2_img(RGB_feature))
      RGB_feature2 = h2 + RGB_feature
      RGB_feature2 = self.relu_RU_end4_img(RGB_feature2)

      ## RU Depth
      Depth_feature = d2
      Depth_feature = self.relu_RU4_depth(Depth_feature)
      Depth_feature = self.relu_RU4_1_depth(self.bn_RU4_1_depth(self.conv_RU4_1_depth(Depth_feature)))
      Depth_feature = self.bn_RU4_2_depth(self.conv_RU4_2_depth(Depth_feature))
      Depth_feature2 = d2 + Depth_feature
      Depth_feature2 = self.relu_RU_end4_depth(Depth_feature2)

      ## out
      RGB_feature2 = torch.mul(gate_out2,RGB_feature2)
      Depth_feature2 = torch.mul(gate_out2, Depth_feature2)
      feature2 = torch.cat((RGB_feature2,Depth_feature2),1)
      feature2 = self.upsample2(feature2)


        ## Level Prediction
      prediction2 = self.pred_ori2(feature2)
      prediction2_L = prediction2
      prediction2 = self.prediction2(prediction2)


      feature2 = self.tran2(feature2)




      ##--------------------------Level 5 ---------------------------------##
      ## GFU
      cat_feature1 = torch.cat((h1, d1), 1)

      cat_feature1 = self.bn5(self.gate_rear5(cat_feature1))

      cat_feature1 = cat_feature1 + feature2

      cat_feature1 = self.relu5_3(cat_feature1)
      cat_feature1_in = self.sconv_relu5(self.sconv_bn5(self.sconv5(cat_feature1)))
      gate_out1 = torch.sigmoid(self.cat_feature_out5(cat_feature1_in))



      ## RU RGB
      RGB_feature = h1
      RGB_feature = self.relu_RU5_img(RGB_feature)
      RGB_feature = self.relu_RU5_1_img(self.bn_RU5_1_img(self.conv_RU5_1_img(RGB_feature)))
      RGB_feature = self.bn_RU5_2_img(self.conv_RU5_2_img(RGB_feature))
      RGB_feature1 = h1 + RGB_feature
      RGB_feature1 = self.relu_RU_end5_img(RGB_feature1)

      ## RU Depth
      Depth_feature = d1
      Depth_feature = self.relu_RU5_depth(Depth_feature)
      Depth_feature = self.relu_RU5_1_depth(self.bn_RU5_1_depth(self.conv_RU5_1_depth(Depth_feature)))
      Depth_feature = self.bn_RU5_2_depth(self.conv_RU5_2_depth(Depth_feature))
      Depth_feature1 = d1 + Depth_feature
      Depth_feature1 = self.relu_RU_end5_depth(Depth_feature1)






        ## Level Prediction
      RGB_feature1 = torch.mul(gate_out1,RGB_feature1)
      Depth_feature1 = torch.mul(gate_out1, Depth_feature1)
      prediction1 = self.pred_ori1(torch.cat((RGB_feature1,Depth_feature1),1))
      prediction1_L = prediction1
      prediction1 = self.prediction1(prediction1)




      prediction5_L = F.upsample(prediction5_L, size=[256, 256], mode='bilinear')
      prediction4_L = F.upsample(prediction4_L, size=[256, 256], mode='bilinear')
      prediction3_L = F.upsample(prediction3_L, size=[256, 256], mode='bilinear')

      f_sum = torch.cat((prediction5_L, prediction4_L, prediction3_L, prediction2_L, prediction1_L), 1)
      feature = self.relu1_m(self.bn1_m(self.conv1_m(f_sum)))
      feature = self.relu2_m(self.bn2_m(self.conv2_m(feature)))
      weight = self.vector_m(self.pre_m(feature))
      weight = F.softmax(weight, dim=1)
      w5, w4, w3, w2, w1 = torch.chunk(weight, 5, 1)
      prediction = w5 * prediction5_L + w4 * prediction4_L + w3 * prediction3_L + w2 * prediction2_L + w1 * prediction1_L
      prediction = self.prediction(prediction)


      return prediction5,prediction4,prediction3,prediction2,prediction1,prediction

























