import {
  ButtonHTMLAttributes,
  DetailedHTMLProps,
  InputHTMLAttributes,
  useCallback,
  useState,
} from "react";
import { PLACEHOLDER_IMAGE_URL } from "./config";
import { CompositeImage } from "./lib/components/CompositeImage";
import { generateQRCodeDataURL } from "./lib/qrcode";
import { FC } from "./lib/types";
import { downloadImage } from "./lib/download";

function App() {
  const [prompt, setPrompt] = useState(``);
  const [qrCodeValue, setQRCodeValue] = useState(``);
  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);

  const generate = useCallback(async () => {
    const dataURL = await generateQRCodeDataURL();
    setQRCodeDataURL(dataURL);
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
        <Button onClick={generate}>GENERATE</Button>
      </UserInput>
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

const Container: FC = ({ children }) => (
  <div className="h-full flex flex-col items-center justify-center">
    {children}
  </div>
);

const UserInput: FC = ({ children }) => <div className="">{children}</div>;

const Input: FC<
  DetailedHTMLProps<InputHTMLAttributes<HTMLInputElement>, HTMLInputElement>
> = ({ ...inputProps }) => (
  <input
    className="w-full border border-gray-500 rounded py-2 px-4 mt-4 first:mt-0"
    {...inputProps}
  />
);

const Button: FC<
  DetailedHTMLProps<ButtonHTMLAttributes<HTMLButtonElement>, HTMLButtonElement>
> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 bg-blue-600 text-white rounded py-2 px-4"
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
