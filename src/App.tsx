import { wait } from "@laxels/utils";
import { useCallback, useState } from "react";
import { PLACEHOLDER_IMAGE_URL } from "./config";
import { CompositeImage } from "./lib/components/CompositeImage";
import { Loader } from "./lib/components/Loader";
import { downloadImage } from "./lib/download";
import { generateQRCodeDataURL } from "./lib/qrcode";
import { ButtonProps, DivProps, FC, FormProps, InputProps } from "./lib/types";

function App() {
  const [prompt, setPrompt] = useState(``);
  const [qrCodeValue, setQRCodeValue] = useState(``);
  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const generate = useCallback(async () => {
    setLoading(true);

    const dataURL = await generateQRCodeDataURL();

    // TODO: fetch from backend
    await wait(5000);

    setQRCodeDataURL(dataURL);
    setLoading(false);
  }, []);

  const downloadQRCode = useCallback(async () => {
    if (!qrCodeDataURL) {
      return;
    }
    await downloadImage({
      url: qrCodeDataURL,
      fileName: `qr-code`,
    });
  }, [qrCodeDataURL]);

  const downloadGeneratedImage = useCallback(async () => {
    await downloadImage({
      url: PLACEHOLDER_IMAGE_URL,
      fileName: `image`,
    });
  }, []);

  return (
    <Container>
      <UserInput>
        <Input
          placeholder={`Image prompt`}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <Input
          placeholder={`QR Code value`}
          value={qrCodeValue}
          onChange={(e) => setQRCodeValue(e.target.value)}
        />
        <Button disabled={loading} onClick={generate}>
          GENERATE
        </Button>
      </UserInput>
      {loading && <Loader />}
      {qrCodeDataURL && (
        <>
          <CompositeImage
            imgSrc={PLACEHOLDER_IMAGE_URL}
            qrCodeDataURL={qrCodeDataURL}
          />
          <DownloadButtons>
            <Button onClick={downloadQRCode}>Download QR Code</Button>
            <Button onClick={downloadGeneratedImage}>Download image</Button>
          </DownloadButtons>
        </>
      )}
    </Container>
  );
}

export default App;

const Container: FC<DivProps> = ({ children }) => (
  <div className="h-full flex flex-col items-center justify-center">
    {children}
  </div>
);

const UserInput: FC<FormProps> = ({ children }) => (
  <form className="" onSubmit={(e) => e.preventDefault()}>
    {children}
  </form>
);

const Input: FC<InputProps> = ({ ...inputProps }) => (
  <input
    className="w-full border border-gray-500 rounded py-2 px-4 mt-4 first:mt-0"
    {...inputProps}
  />
);

const Button: FC<ButtonProps> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 disabled:bg-gray-300 bg-blue-600 text-white rounded py-2 px-4"
    {...buttonProps}
  />
);

const DownloadButtons: FC = ({ children }) => (
  <div className="">{children}</div>
);

// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
// const Container: FC = ({ children }) => (
//   <div className="w-full">{children}</div>
// );
