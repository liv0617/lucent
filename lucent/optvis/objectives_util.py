# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for Objectives."""

from __future__ import absolute_import, division, print_function

import torch

def _dot_cossim(x, y, cossim_pow):
  eps = 1e-4
  x = torch.squeeze(x)
  y = torch.squeeze(y)
  xy_dot = torch.dot(x, y)
  if cossim_pow == 0: return xy_dot.mean()
  x_mags = torch.sqrt(torch.dot(x,x))
  y_mags = torch.sqrt(torch.dot(y,y))
  cossims = xy_dot / (eps + x_mags ) / (eps + y_mags)
  floored_cossims = cossims.clamp(0.1)
  return (xy_dot * floored_cossims**cossim_pow).mean()


def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(acts, x=None, y=None):
    shape = acts.shape
    x = shape[2] // 2 if x is None else x
    y = shape[3] // 2 if y is None else y
    return acts[:, :, y:y+1, x:x+1]


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2

layer_branches = {
    "mixed3a": {"1x1": 64, "3x3": 128, "5x5": 32, "pool_reduce": 32},
    "mixed3b": {"1x1": 128, "3x3": 192, "5x5": 96, "pool_reduce": 64},
    "mixed4a": {"1x1": 192, "3x3": 204, "5x5": 48, "pool_reduce": 64},
    "mixed4b": {"1x1": 160, "3x3": 224, "5x5": 64, "pool_reduce": 64},
    "mixed4c": {"1x1": 128, "3x3": 256, "5x5": 64, "pool_reduce": 64},
    "mixed4d": {"1x1": 112, "3x3": 288, "5x5": 64, "pool_reduce": 64},
    "mixed4e": {"1x1": 256, "3x3": 320, "5x5": 128, "pool_reduce": 128},
    "mixed5a": {"1x1": 256, "3x3": 320, "5x5": 128, "pool_reduce": 128},
    "mixed5b": {"1x1": 384, "3x3": 384, "5x5": 128, "pool_reduce": 128},
}
