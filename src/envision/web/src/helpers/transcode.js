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
import FFmpeg from "@ffmpeg/ffmpeg";

// Transcode to MP4
export default async function transcode(blob, onMessage = (message) => {}) {
  onMessage("Loading transcoding library (FFMPEG)");
  const { createFFmpeg } = FFmpeg;
  const ffmpeg = createFFmpeg({ log: true });
  await ffmpeg.load();

  onMessage("Transcoding video to MP4");
  let buffer = new Uint8Array(await blob.arrayBuffer());

  ffmpeg.FS("writeFile", "input.webm", buffer);
  await ffmpeg.run(
    "-i",
    "input.webm",
    // XXX: We can do this because the RecordRTC is giving us h264 encoded data
    //      (https://github.com/muaz-khan/RecordRTC#configuration) all we need to do
    //      is switch the container.
    "-c:v",
    "copy",
    "output.mp4"
  );
  onMessage("Transcoding complete");

  let output = ffmpeg.FS("readFile", "output.mp4");
  let outputBlog = new Blob([output.buffer], { type: "video/mp4" });
  return outputBlog;
}
