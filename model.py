import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

class AutoencoderWithClassification(nn.Module):
    def __init__(self, num_classes):
        super(AutoencoderWithClassification, self).__init__()
        
        # Load ResNet-50 without classification layers
        resnet = models.resnet50(pretrained=True)
        
        # Modify first conv layer to accept 1 channel input
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.encoder_rest = nn.Sequential(*list(resnet.children())[4:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.encoder_rest(x)
        
        # Decoder
        decoded_output = self.decoder(x)
        second_latent = self.encoder_rest(self.encoder(decoded_output))
        
        # Classification
        classification_output = self.classification_head(x)
        
        return decoded_output, classification_output, x, second_latent

# Create an instance of the model
# num_classes = 10  # Adjust based on your dataset
# model = AutoencoderWithClassification(num_classes)


# # summary(model=model, 
# #         input_size=(1, 1, 512, 512), # make sure this is "input_size", not "input_shape"
# #         # col_names=["input_size"], # uncomment for smaller output
# #         col_names=["input_size", "output_size", "num_params", "trainable"],
# #         col_width=20,
# #         row_settings=["var_names"]
# # )

# # Testing the model with random input
# input_tensor = torch.randn(1, 1, 512, 512)  # 1 channel input with size 512x512
# decoded_output, classification_output = model(input_tensor)

# print("Decoded Output Shape:", decoded_output.shape)
# print("Classification Output Shape:", classification_output.shape)
