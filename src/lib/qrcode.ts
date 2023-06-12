import QRCode from "qrcode";
import { BACKEND_URL } from "../config";

export async function generateQRCodeDataURL(
  str: string
): Promise<string | null> {
  try {
    const res = await QRCode.toDataURL(str);
    return res;
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
  }
  return null;
}

export async function generateImage(prompt: string, dataURL: string): Promise<string | null> {
  try {
    // Create job
    const jobResponse = await fetch(`${BACKEND_URL}/job/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ prompt: prompt, image: { image_data: dataURL } })
    });
    const jobData = await jobResponse.json();
    const jobId = jobData.job_id;

    // Check job status
    let jobStatus = 'NOT_STARTED';
    while (jobStatus !== 'COMPLETE' && jobStatus !== 'FAILED') {
      await new Promise(r => setTimeout(r, 1000));  // Wait for a second
      const statusResponse = await fetch(`${BACKEND_URL}/job/?job_id=${jobId}`);
      const statusData = await statusResponse.json();
      jobStatus = statusData.status;
    }

    // If job failed, return null
    if (jobStatus === 'FAILED') {
      console.error(`Job failed: ${jobId}`);
      return null;
    }

    // Get job result
    const resultResponse = await fetch(`${BACKEND_URL}/job/${jobId}`);
    return URL.createObjectURL(await resultResponse.blob());
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
    return null;
  }
}
