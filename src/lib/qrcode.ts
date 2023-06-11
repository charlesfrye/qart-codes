import QRCode from "qrcode";
import { BACKEND_URL } from "../config";

export async function generateQRCodeDataURL(): Promise<string | null> {
  try {
    const res = await QRCode.toDataURL("I am a pony!");
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
