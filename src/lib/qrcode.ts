import QRCode from "qrcode";

export async function generateQRCodeDataURL(): Promise<string | null> {
  try {
    const res = await QRCode.toDataURL("I am a pony!");
    return res;
  } catch (err) {
    console.error(`Error generating QR code: ${err}`);
  }
  return null;
}
