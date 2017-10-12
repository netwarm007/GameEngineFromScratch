#!/bin/bash
pip install glad
glad --generator=c --spec gl --out-path=.
glad --generator=c --spec wgl --out-path=.
glad --generator=c --spec glx --out-path=.
