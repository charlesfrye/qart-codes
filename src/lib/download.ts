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

    // Create an anchor element
    const link = document.createElement("a");

    // Set the href attribute to the image Blob URL
    link.href = URL.createObjectURL(blob);

    // Set the download attribute to the desired image name with extension
    link.download = `${fileName}.png`;

    // Append the link to the document
    document.body.appendChild(link);

    // Trigger the click event to start the download
    link.click();

    // Remove the link from the document and revoke the Blob URL after the download is initiated
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  } catch (err) {
    console.error("Failed to download the image:", err);
  }
}
