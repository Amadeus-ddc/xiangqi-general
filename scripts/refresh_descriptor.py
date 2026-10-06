"""Regenerate the pinned Px0 Python descriptor without a protoc/runtime lock."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="vendor/px0/proto/net.proto")
    args = parser.parse_args()
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    path = Path(args.proto)
    with tempfile.TemporaryDirectory() as temporary:
        descriptor = Path(temporary) / "net.desc"
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{path.parent}",
                        f"--descriptor_set_out={descriptor}", str(path)], check=True)
        value = FileDescriptorSet.FromString(descriptor.read_bytes()).file[0].SerializeToString()
    content = ('# Px0 protocol, GPL-3.0-or-later; Copyright (C) The LCZero/Px0 Authors.\n'
               'from google.protobuf import descriptor_pool\nfrom google.protobuf.internal import builder\n'
               f'DESCRIPTOR = descriptor_pool.Default().AddSerializedFile({value!r})\n'
               'builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n'
               'builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n')
    Path("src/xqgeneral/net_pb2.py").write_text(content)


if __name__ == "__main__":
    main()
