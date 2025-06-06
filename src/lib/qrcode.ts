import QRCode from "qrcode";
import { BACKEND_URL, QR_CODE_DIMENSIONS } from "../config";

type QRCodeOptions = QRCode.QRCodeOptions & {
  // Missing properties in the type definition
  width?: number;
};

export async function generateQRCodeDataURL(
  str: string,
  options: QRCodeOptions = {
    errorCorrectionLevel: "H",
    width: QR_CODE_DIMENSIONS,
  }
): Promise<string | null> {
  try {
    const res = await QRCode.toDataURL(str, options);
    return res;
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
  }
  return null;
}

type JobStatus =
  | `NOT_STARTED`
  | `PENDING`
  | `RUNNING`
  | `COMPLETE`
  | `CONSUMED`
  | `CANCELLED`
  | `FAILED`
  | `UNKNOWN`;

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

interface JsonResponseEntry {
  evaluation?: {
    detected: boolean;
    aesthetic_rating: number;
  };
  image: string;
}

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
    const jsonResponse: JsonResponseEntry[] = await resultResponse.json();

    if (jsonResponse?.length > 0) {
      // Filter for detected codes
      const detectedGenerations =
        jsonResponse.some((gen) => gen.evaluation?.detected)
          ? jsonResponse.filter((gen) => gen.evaluation?.detected)
          : jsonResponse;

      // Find the entry with the highest 'aesthetic_rating'
      const bestGen = detectedGenerations.reduce(
        (prev, current) =>
          (current.evaluation?.aesthetic_rating ?? 0) >
          (prev.evaluation?.aesthetic_rating ?? 0)
            ? current
            : prev,
        jsonResponse[0]
      );

      const base64Image = bestGen.image;
      return {
        status,
        result: `data:image/png;base64,${base64Image}`,
      };
    } else throw new Error("Invalid API response: payload is missing or empty");
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
