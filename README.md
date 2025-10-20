## HuggingFace Spaces Containers for ARM64 (and X64)

Looking for only X64 containers? Just use the ones from HuggingFace Spaces directly and don't waste your time.

For my own purposes, converting some HuggingFace Spaces to be compatible with ARM64 since they only host X64 containers.

X64 containers are just built for convenience to only need one endpoint.

The image name is the HF space name all lowercased, located at `ghcr.io/eadwu`. The tags for use are simple, `:ARM64` and `:X64`.

If you don't use the tag, you will *__typically__* find that pulling will point you to a multi-arch manifest.

| HF Space | ghcr.io |
|--|--|
| [IndexTTS-2-Demo](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo) | [ghcr.io/eadwu/indextts-2-demo](https://github.com/eadwu/hf-spaces/pkgs/container/indextts-2-demo) |
| [higgs_audio_v2](https://huggingface.co/spaces/smola/higgs_audio_v2) | [ghcr.io/eadwu/higgs_audio_v2](https://github.com/eadwu/hf-spaces/pkgs/container/higgs_audio_v2) |
| [neutts-air](https://huggingface.co/spaces/neuphonic/neutts-air) | [ghcr.io/eadwu/neutts-air](https://github.com/eadwu/hf-spaces/pkgs/container/neutts-air) |

