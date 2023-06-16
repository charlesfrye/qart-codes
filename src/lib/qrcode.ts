import QRCode from "qrcode";
import { BACKEND_URL } from "../config";

export async function generateQRCodeDataURL(
  str: string,
  options: QRCode.QRCodeOptions = { errorCorrectionLevel: "H" }
): Promise<string | null> {
  try {
    const res = await QRCode.toDataURL(str, options);
    return res;
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
  }
  return null;
}

type JobStatus = `NOT_STARTED` | `RUNNING` | `COMPLETE` | `FAILED`;

export async function startGeneration(
  prompt: string,
  dataURL: string
): Promise<string | null> {
  try {
    // Create job
    const jobResponse = await fetch(`${BACKEND_URL}/job`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: prompt, image: { image_data: dataURL } }),
    });
    const jobData = await jobResponse.json();
    return jobData.job_id;
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
    return null;
  }
}

type PollResult = {
  status: JobStatus;
  result?: string;
};

export async function pollGeneration(jobID: string): Promise<PollResult> {
  const statusResponse = await fetch(`${BACKEND_URL}/job?job_id=${jobID}`);
  const statusData = await statusResponse.json();
  const status: JobStatus = statusData.status;

  // If job failed, return null
  if (status === `FAILED`) {
    console.error(`Job failed: ${jobID}`);
    return { status };
  }

  if (status === `COMPLETE`) {
    // Get job result
    const resultResponse = await fetch(`${BACKEND_URL}/job/${jobID}`);
    return { status, result: URL.createObjectURL(await resultResponse.blob()) };
  }

  return { status };
}

export async function cancelGeneration(jobID: string): Promise<void> {
  try {
    await fetch(`${BACKEND_URL}/job/${jobID}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });
  } catch (err) {
    console.error(`Error cancelling job: ${err}`);
  }
}
