// MIT License
//
// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
export const Red = [210 / 255, 30 / 255, 30 / 255, 1];
export const Rose = [196 / 255, 0, 84 / 255, 1];
export const Maroon = [128 / 255, 0, 0, 1];
export const Orange = [237 / 255, 109 / 255, 0, 1];
export const Yellow = [255 / 255, 190 / 255, 40 / 255, 1];
export const GreenTransparent = [98 / 255, 178 / 255, 48 / 255, 0.3];
export const Silver = [192 / 255, 192 / 255, 192 / 255, 1];
export const Black = [0, 0, 0, 1];
export const Green = [30 / 255, 210 / 255, 30 / 255, 1];

export const DarkBlue = [5 / 255, 5 / 255, 70 / 255, 1];
export const Blue = [0, 153 / 255, 1, 1];
export const LightBlue = [173 / 255, 216 / 255, 230 / 255, 1];
export const BlueTransparent = [60 / 255, 170 / 255, 200 / 255, 0.6];

export const DarkCyan = [47 / 255, 79 / 255, 79 / 255, 1];
export const CyanTransparent = [48 / 255, 181 / 255, 197 / 255, 0.5];

export const DarkPurple = [50 / 255, 30 / 255, 50 / 255, 1];
export const Purple = [127 / 255, 0, 127 / 255, 1];

export const DarkGrey = [80 / 255, 80 / 255, 80 / 255, 1];
export const Grey = [119 / 255, 136 / 255, 153 / 255, 1];
export const LightGreyTransparent = [221 / 255, 221 / 255, 221 / 255, 0.1];

export const OffWhite = [200 / 255, 200 / 255, 200 / 255, 1];
export const White = [1, 1, 1, 1];

export const SceneColors = Object.freeze({
  Agent: Red,
  SocialAgent: Silver,
  SocialVehicle: Silver,
  Road: DarkGrey,
  Selection: LightBlue,
  Interest: Blue,
  EgoWaypoint: CyanTransparent,
  EgoDrivenPath: CyanTransparent,
  BubbleLine: LightGreyTransparent,
  MissionRoute: GreenTransparent,
  LaneDivider: OffWhite,
  EdgeDivider: Yellow,
});

export const SignalColors = Object.freeze({
  Unknown: Grey,
  Stop: Maroon,
  Caution: Yellow,
  Go: Green,
});
