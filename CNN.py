import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):          # 默认 10 类，可改
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),      # 3×32×32 -> 16×32×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 16×16×16

            nn.Conv2d(16, 32, 3, padding=1),     # 32×16×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                      # 32×8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 32*8*8 = 2048
            nn.Linear(32*8*8, num_classes)       # 直接输出 logits
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# model = SimpleCNN(num_classes=10)