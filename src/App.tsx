import { useCallback, useState } from "react";
import { CompositeImage } from "./lib/components/CompositeImage";
import { Loader } from "./lib/components/Loader";
import { downloadImage } from "./lib/download";
import { generateImage, generateQRCodeDataURL } from "./lib/qrcode";
import { ButtonProps, DivProps, FC, FormProps, InputProps } from "./lib/types";

function App() {
  const [prompt, setPrompt] = useState(``);
  const [qrCodeValue, setQRCodeValue] = useState(``);
  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const generate = useCallback(async () => {
    setLoading(true);

    const dataURL = await generateQRCodeDataURL();

    // TODO: fetch from backend
    const generatedSrc = await generateImage();

    setQRCodeDataURL(dataURL);
    setImgSrc(generatedSrc);
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
    if (!imgSrc) {
      return;
    }
    await downloadImage({
      url: imgSrc,
      fileName: `image`,
    });
  }, [imgSrc]);

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
          Generate
        </Button>
      </UserInput>
      {(loading || (imgSrc && qrCodeDataURL)) && (
        <ResultsContainer>
          {loading && <Loader />}
          {imgSrc && qrCodeDataURL && (
            <>
              <CompositeImage imgSrc={imgSrc} qrCodeDataURL={qrCodeDataURL} />
              <DownloadButtons>
                <Button onClick={downloadQRCode}>Download QR Code</Button>
                <Button onClick={downloadGeneratedImage}>Download image</Button>
              </DownloadButtons>
            </>
          )}
        </ResultsContainer>
      )}
    </Container>
  );
}

export default App;

const Container: FC<DivProps> = ({ children }) => (
  <div className="min-h-full flex flex-col items-center justify-center p-4 bg-blue">
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
    className="w-full border border-gray-500 rounded-xl py-2.5 px-8 mt-4 first:mt-0"
    {...inputProps}
  />
);

const Button: FC<ButtonProps> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 disabled:bg-gray-300 bg-orange hover:bg-orange-light text-white rounded-xl py-2.5 px-8 transition-colors font-bold"
    {...buttonProps}
  />
);

const ResultsContainer: FC<DivProps> = ({ children }) => (
  <div className="mt-16">{children}</div>
);

const DownloadButtons: FC = ({ children }) => (
  <div className="">{children}</div>
);
