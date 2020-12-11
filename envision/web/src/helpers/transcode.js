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
