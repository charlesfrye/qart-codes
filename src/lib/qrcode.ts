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

export async function generateImage(): Promise<string | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/job/_test`);
    return URL.createObjectURL(await res.blob());
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
  }
  return null;
}
