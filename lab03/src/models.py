
import torch
import torch.nn as nn


class MLP(nn.Module):
   

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
   

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class AttentionModel(nn.Module):
    

    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 4, 7, 4, 7)

        
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, 16, 49)

        x = self.patch_projection(x)
        x = self.encoder(x)

        
        x = x.mean(dim=1)
        return self.classifier(x)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    dummy_input = torch.randn(2, 1, 28, 28)

    for name, model in models.items():
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"{name}:")
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  Output shape: {tuple(output.shape)}")
