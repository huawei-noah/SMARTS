# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import re
import subprocess


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def git_most_recent_tag(repo_path: str = None, matching_regex: str = None) -> str:
    path = _normalize_path(repo_path) if repo_path else None
    try:
        tagstr = subprocess.check_output(["git", "tag"], cwd=path).decode().strip("\n")
    except:
        return None
    if not tagstr:
        return None
    pat = re.compile(matching_regex) if matching_regex else None
    tags = tagstr.split("\n")
    tagdate = {}
    for tag in tags:
        if pat and not pat.search(tag):
            continue
        base = (
            subprocess.check_output(["git", "merge-base", "HEAD", tag], cwd=path)
            .decode()
            .strip("\n")
        )
        if base:
            tagdate[tag] = subprocess.check_output(
                ["git", "log", "-n", "1", base, "--format=%at"], cwd=path
            ).decode()
    return (
        sorted(tagdate.items(), key=lambda ti: ti[1] + ti[0], reverse=True)[0][0]
        if tagdate
        else None
    )


def git_version(repo_path: str = None, ver_prefix: str = "v") -> str:
    tag_regex = "^" + ver_prefix if ver_prefix else None
    tag = git_most_recent_tag(repo_path, tag_regex)
    return tag[len(ver_prefix) :] if ver_prefix else tag
