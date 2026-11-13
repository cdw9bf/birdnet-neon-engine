"""bnc: the bnneon model compiler (ONNX -> .bnpk model package).

Modules: bnc (the command line), onnxg (ONNX import), canon (exporter artifact
rewrites), recog (the recognizer: ONNX nodes -> IR ops), lower (IR -> kernels
per CPU path), pkgfmt (the .bnpk writer) and errs (errors and exit codes).
"""
