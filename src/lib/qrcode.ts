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
  results?: string[];
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
    const payload: JsonResponseEntry[] = await resultResponse.json();

    if (!Array.isArray(payload) || payload.length === 0) {
      throw new Error("Invalid API response: payload is missing or empty");
    }

    // return in the order best for display:
    const orderedResults = [...payload].sort((a, b) => {
      const detA = !!a.evaluation?.detected;
      const detB = !!b.evaluation?.detected;
      if (detA !== detB) return detA ? -1 : 1;            // scannable before not

      const rA = a.evaluation?.aesthetic_rating ?? -Infinity;
      const rB = b.evaluation?.aesthetic_rating ?? -Infinity;
      return rB - rA;                                     // higher ratings first
    });


      return {
        status,
        results: orderedResults.map(({ image }) => `data:image/png;base64,${image}`)
      };
    }

  return { status }
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
