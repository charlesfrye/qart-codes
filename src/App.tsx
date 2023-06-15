import { useCallback, useState, useRef, useEffect } from "react";
import { toast } from "react-toastify";
import { CompositeImage } from "./lib/components/CompositeImage";
import { Loader } from "./lib/components/Loader";
import { createDivContainer } from "./lib/components/helpers";
import { downloadImage } from "./lib/download";
import { generateImage, generateQRCodeDataURL } from "./lib/qrcode";
import { ButtonProps, FC, FormProps, InputProps, TextareaProps } from "./lib/types";

function App() {
  const [prompt, setPrompt] = useState(`fireworks, beautiful display above a city, breathtaking spectacle, fiery, shimmering, symphonic, cascading`);
  const [qrCodeValue, setQRCodeValue] = useState(`tfs.ai/qr-main`);
  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const generate = useCallback(async () => {
    if (!prompt) {
      toast(`Please enter a text prompt`);
      return;
    }
    if (!qrCodeValue) {
      toast(`Please enter a QR code value`);
      return;
    }

    if (qrCodeValue.length > 25 && !localStorage.getItem('warnedAboutLongQRCode')) {
      localStorage.setItem('warnedAboutLongQRCode', 'true');
      toast.warn(`Yo, Q-Art Codes work better with shorter text. Try a URL shortener, leave off http and www. KISS!`);
    }

    setLoading(true);

    const dataURL = await generateQRCodeDataURL(qrCodeValue);

    if (!dataURL) {
      console.error("error generating QR code")
      return;
    }

    const generatedSrc = await generateImage(prompt, dataURL);
    setLoading(false);
    if (!generatedSrc) {
      toast(`Ah geez, something borked. Try again, it'll probably be faster!`);
      return;
    }

    setQRCodeDataURL(dataURL);
    setImgSrc(generatedSrc);
  }, [prompt, qrCodeValue]);

  const downloadQRCode = useCallback(async () => {
    if (!qrCodeDataURL) {
      return;
    }
    await downloadImage({
      url: qrCodeDataURL,
      fileName: `qr-code`,
    });
  }, [qrCodeDataURL]);

  const downloadQArtCode = useCallback(async () => {
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
        <Textarea
          placeholder={`Visual content or style to apply to the QR code`}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <Input
          placeholder={`Text to encode, like a URL or your wifi password`}
          value={qrCodeValue}
          onChange={(e) => setQRCodeValue(e.target.value)}
        />
        <Button disabled={loading} onClick={generate}>
          Generate Q-Art Code
        </Button>
      </UserInput>
      {(loading || (imgSrc && qrCodeDataURL)) && (
        <ResultsContainer>
          {loading && <Loader />}
          {imgSrc && qrCodeDataURL && (
            <>
              <CompositeImage imgSrc={imgSrc} qrCodeDataURL={qrCodeDataURL} />
              <DownloadButtons>
                <Button onClick={downloadQArtCode}>Download Q-Art Code</Button>
                <Button onClick={downloadQRCode}>Download QR Code</Button>
              </DownloadButtons>
            </>
          )}
        </ResultsContainer>
      )}
    </Container>
  );
}

export default App;

const Container = createDivContainer(
  "min-h-full flex flex-col items-center justify-center p-4 bg-blue"
);

const UserInput: FC<FormProps> = ({ children }) => (
  <form onSubmit={(e) => e.preventDefault()}>{children}</form>
);

const Input: FC<InputProps> = ({ ...inputProps }) => (
  <input
    className="w-full border border-gray-500 rounded-xl py-2.5 px-8 mt-4 first:mt-0 focus-visible:outline-none"
    {...inputProps}
  />
);

const Textarea: FC<TextareaProps> = ({ ...inputProps }) => {
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = "auto";
      textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
    }
  }, [inputProps.value]);

  return (
    <textarea
      ref={textAreaRef}
      className="w-full border border-gray-500 rounded-xl py-2.5 px-8 mt-4 first:mt-0 focus:outline-none resize-none overflow-hidden max-h-60"
      {...inputProps}
    />
  );
};

const Button: FC<ButtonProps> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 disabled:bg-gray-300 bg-orange hover:bg-orange-light text-white rounded-xl py-2.5 px-8 transition-colors font-bold"
    {...buttonProps}
  />
);

const ResultsContainer = createDivContainer("mt-16");

const DownloadButtons = createDivContainer(``);
