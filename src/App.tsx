import { useCallback, useState, useRef, useEffect } from "react";
import { toast } from "react-toastify";
import { CompositeImage } from "./lib/components/CompositeImage";
import { Loader } from "./lib/components/Loader";
import { createDivContainer } from "./lib/components/helpers";
import { downloadImage } from "./lib/download";
import {
  cancelGeneration,
  startGeneration,
  generateQRCodeDataURL,
  pollGeneration,
} from "./lib/qrcode";
import {
  ButtonProps,
  FC,
  FormProps,
  InputProps,
  TextareaProps,
} from "./lib/types";
import { wait } from "@laxels/utils";

function App() {
  const [prompt, setPrompt] = useState(
    `fireworks, beautiful display above a city, breathtaking spectacle, fiery, shimmering, symphonic, cascading`
  );
  const [qrCodeValue, setQRCodeValue] = useState(`tfs.ai/qr-main`);

  const [loading, setLoading] = useState(false);
  const [jobID, setJobID] = useState<string | null>(null);
  const cancelledRef = useRef(false);

  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);

  const generate = useCallback(async () => {
    if (!prompt) {
      toast(`Please enter a text prompt`);
      return;
    }
    if (!qrCodeValue) {
      toast(`Please enter a QR code value`);
      return;
    }

    if (
      qrCodeValue.length > 25 &&
      !localStorage.getItem("warnedAboutLongQRCode")
    ) {
      localStorage.setItem("warnedAboutLongQRCode", "true");
      toast.warn(
        `Yo, Q-Art Codes work better with shorter text. Try a URL shortener, leave off http and www. KISS!`
      );
    }

    setLoading(true);
    cancelledRef.current = false;

    const dataURL = await generateQRCodeDataURL(qrCodeValue);

    if (!dataURL) {
      console.error("error generating QR code");
      setLoading(false);
      return;
    }

    const handleGenerationFailure = (waitingTime: number) => {
      const waitColdBoot = 60_000; // typical waiting time when backend hits a cold boot
      let message = "Ah geez, something borked. Try again.";
      if (waitingTime >= waitColdBoot) {
        message += " It'll probably be faster!";
      }
      toast(message);
      setLoading(false);
    };

    const jobID = await startGeneration(prompt, dataURL);
    if (!jobID) {
      handleGenerationFailure(-1);
      return;
    }
    setJobID(jobID);

    const start = Date.now();
    let waitingTime = 0;
    const maxWaiting = 300_000; // set defensively high, backend should timeout first
    let waitedTooLong = false;
    while (!waitedTooLong) {
      const pollInterval = 1_000;
      await wait(pollInterval);
      if (cancelledRef.current) {
        break;
      }

      const { status, result } = await pollGeneration(jobID);
      waitingTime = Date.now() - start;

      if (waitingTime >= maxWaiting) {
        waitedTooLong = true;
      }

      if (status === `FAILED`) {
        handleGenerationFailure(waitingTime);
        break;
      }

      if (status === `COMPLETE` && result) {
        setQRCodeDataURL(dataURL);
        setImgSrc(result);
        break;
      }
    }
    if (waitedTooLong) {
      handleGenerationFailure(waitingTime);
    }
    setLoading(false);
  }, [prompt, qrCodeValue]);

  const cancel = useCallback(async () => {
    if (!loading || !jobID) {
      return;
    }
    cancelGeneration(jobID);
    setJobID(null);
    setLoading(false);
    cancelledRef.current = true;
  }, [loading, jobID]);

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
        {!loading ? (
          <Button disabled={loading} onClick={generate}>
            Generate Q-Art Code
          </Button>
        ) : (
          <Button onClick={cancel}>Cancel Generation</Button>
        )}
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

const ResultsContainer = createDivContainer(`mt-16 w-full max-w-[512px]`);

const DownloadButtons = createDivContainer(``);
