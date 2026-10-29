"""No built-in AscendC baseline: the experiment supplies and freezes it separately."""


def build(**kwargs):
    raise NotImplementedError("SSD device baseline is supplied by the external AscendC comparison")
