"""Additional official sources/checkpoints, pinned and verified by the shared resolver."""

from src.models.pretrained_assets import ASSETS

PHYSIOEX_REVISION = "66b24320aa39bc635e304b0891eff41131ca54a5"
PHYSIOEX_CHECKPOINT_REVISION = "fc362d1976a203f9bf47cd5072845615d573aae2"

SERIES_ASSETS = {
    "anysleep-fusion-components.py": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/base/model/usleep_base_components.py",
        "84a1d1ed7df9d515d0b9d7bccaaaefa2ab7a36e165aa6a289438b5ed991b7b1f",
    ),
    "anysleep-early-fusion.py": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/base/model/anysleep_early_fusion.py",
        "5561f803ded3b6c154d1a766873ab7c774e50f61b8448692c6abd0a5ad3cc9f2",
    ),
    "anysleep-late-fusion.py": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/base/model/anysleep_late_fusion.py",
        "f596bced2a7a9468ed47ad30239f0212e18aad625d20ba1601dbb0b306a5e3a2",
    ),
    "anysleep-fusion-utils.py": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/base/model/utilities.py",
        "a628e4173d8132ebb9fb8f672e7243432cd3394a1cda00f8fb55f5783c50fed5",
    ),
    "anysleep-init.py": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/base/model/base_model.py",
        "a48f9bbe4d4ffad98b47d6bb468746365391fd4ec0aec2d014d14b58098b7ce5",
    ),
    "anysleep-author-split.yaml": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/config/usleep_split.yaml",
        "30327af675916ca7f1263d76c057307fb78a3622d46e27560280719ec11543e1",
    ),
    "anysleep-early-fusion-run1.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-early-fusion-run1.pth",
        "2c7b1500fc03d47cf1191825e3a5963e1de913f184bc884f7a94794c0063674e",
    ),
    "anysleep-early-fusion-run2.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-early-fusion-run2.pth",
        "f80ed97eea2c722e3483b973d90043569ab67876fc88b9ab15c52da0acc5bafa",
    ),
    "anysleep-early-fusion-run3.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-early-fusion-run3.pth",
        "df4c26b91cbf43d63d44d3e6e2b7e2c3d95b3f8c6b8c57f10faff171ce06de2f",
    ),
    "anysleep-late-fusion-run1.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-late-fusion-run1.pth",
        "76916975f545d6ec1fbfe4e4153e02d5cd6f287a20103d1a519821ab71876f00",
    ),
    "anysleep-late-fusion-run2.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-late-fusion-run2.pth",
        "e859f0142731e25cd8225e130e5ab5b24b65a2e8dceb1991da032f3f320943b6",
    ),
    "anysleep-late-fusion-run3.pth": (
        "https://raw.githubusercontent.com/dslaborg/AnySleep/899bfff284f3eb3e7448bc90990f2230757470a2/models/anysleep-late-fusion-run3.pth",
        "ce3275f94cb7f26e2cb89608ea3fa9cecc20cae02329264c5ed77a9cb0258e78",
    ),
    "tinysleepnet-source.py": (
        "https://raw.githubusercontent.com/guidogagl/physioex/66b24320aa39bc635e304b0891eff41131ca54a5/physioex/models/tinysleepnet.py",
        "a19fa56baa6aa66f6d2ba35e57ec5e6368bf7076b84b712670b888ce5294a7f5",
    ),
    "tinysleepnet-config.json": (
        "https://huggingface.co/4rooms/physioex/resolve/fc362d1976a203f9bf47cd5072845615d573aae2/tinysleepnet-supratak/config.json",
        "765544103d4cf2b0dd50681206296e866c0b47b62eeaf4c54e7e2cf5587b0a19",
    ),
    "tinysleepnet.pt": (
        "https://huggingface.co/4rooms/physioex/resolve/fc362d1976a203f9bf47cd5072845615d573aae2/tinysleepnet-supratak/model.pt",
        "f6cf85b7499c91ed6cb665aa5864ea498a4e0d801288e900a0da9c23d3c5337d",
    ),
}

ASSETS.update(SERIES_ASSETS)
