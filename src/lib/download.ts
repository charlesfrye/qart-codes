import { QR_CODE_DIMENSIONS } from "../config";

type DownloadImageParams = {
  url: string;
  fileName: string;
};

export async function downloadImage({
  url,
  fileName,
}: DownloadImageParams): Promise<void> {
  try {
    // Fetch the image data
    const response = await fetch(url);
    const blob = await response.blob();

    const objectURL = URL.createObjectURL(blob);
    downloadImageFromDataURL(objectURL, fileName);

    // Revoke the Blob URL after the download is initiated
    URL.revokeObjectURL(objectURL);
  } catch (err) {
    console.error("Failed to download the image:", err);
  }
}

function downloadImageFromDataURL(dataURL: string, fileName: string): void {
  // Create an anchor element
  const link = document.createElement("a");

  // Set the href attribute to the image Blob URL
  link.href = dataURL;

  // Set the download attribute to the desired image name with extension
  link.download = `${fileName}.png`;

  // Append the link to the document
  document.body.appendChild(link);

  // Trigger the click event to start the download
  link.click();

  // Remove the link from the document
  document.body.removeChild(link);
}

export async function downloadStitchedImage(
  src1: string,
  src2: string,
  leftPortion: number
): Promise<void> {
  try {
    const dataURL = await stitchImages(src1, src2, leftPortion);
    await downloadImageFromDataURL(dataURL, `stitched`);
  } catch (err) {
    console.error("Failed to download stitched image:", err);
  }
}

async function stitchImages(
  src1: string,
  src2: string,
  leftPortion: number
): Promise<string> {
  const d = QR_CODE_DIMENSIONS;
  const rightPortion = 1 - leftPortion;

  const leftWidthUnrounded = leftPortion * d;
  const rightWidthUnrounded = rightPortion * d;
  const leftWidthRounded = Math.round(leftWidthUnrounded);
  const rightWidthRounded = Math.round(rightWidthUnrounded);
  const leftWidth = leftWidthRounded;
  const rightWidth =
    rightWidthRounded -
    (leftWidthRounded > leftWidthUnrounded &&
    rightWidthRounded > rightWidthUnrounded
      ? 1
      : 0);

  const canvas = document.createElement(`canvas`);
  canvas.width = d;
  canvas.height = d;

  const ctx = canvas.getContext("2d");
  if (ctx == null) {
    throw new Error(`Unable to get canvas context`);
  }

  const [img1, img2] = await Promise.all([loadImage(src1), loadImage(src2)]);
  ctx.drawImage(img1, 0, 0, leftWidth, d, 0, 0, leftWidth, d);
  ctx.drawImage(img2, leftWidth, 0, rightWidth, d, leftWidth, 0, rightWidth, d);

  return canvas.toDataURL("image/png");
}

async function loadImage(src: string): Promise<HTMLImageElement> {
  const d = QR_CODE_DIMENSIONS;
  const img = new Image(d, d);
  img.src = src;

  const loadedImg = await new Promise<HTMLImageElement>((resolve) => {
    img.onload = function () {
      resolve(img);
    };
  });

  return loadedImg;
}
