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
    `neon green cubes, rendered in blender, trending on artstation`
  );
  const [qrCodeValue, setQRCodeValue] = useState(`modal.com`);

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
      const waitColdBoot = 90_000; // typical waiting time when backend hits a cold boot
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
			<div className="mb-6">
    <label
      htmlFor="prompt"
      className="block text-24 font-degular font-light text-green-light mb-2"
    >
      Prompt
    </label>
        <Textarea
          placeholder={`Visual content or style to apply to the QR code`}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
				</div>
			<div className="mb-6">
			<label
      htmlFor="qrValue"
      className="block text-24 font-degular font-light text-green-light mb-2"
    >
      Link
    </label>

    <Input
      id="qrValue"
      placeholder="Text to encode, like a URL or your Wi-Fi password"
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
				</div>
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
      <Footer />
    </Container>
  );
}

export default App;

const Container = createDivContainer(
  "min-h-full flex flex-col items-center justify-center p-4 bg-black"
);

const UserInput: FC<FormProps> = ({ children }) => (
  <form onSubmit={(e) => e.preventDefault()}>{children}</form>
);

const Input: FC<InputProps> = ({ ...inputProps }) => (
  <input
    className="w-full rounded-xl py-2.5 px-8 mt-4 first:mt-0
               border border-gray-500
               bg-green-light/10          /* 10 % opacity */
               text-green-light font-degular font-light
               placeholder:text-green-light/60
               focus-visible:outline-none"
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
      className="w-full rounded-xl py-2.5 px-8 mt-4 first:mt-0
                 border border-gray-500
                 bg-green-light/10         /* 10 % opacity */
                 text-green-light font-degular font-light leading-relaxed
                 placeholder:text-green-light/60
                 focus:outline-none resize-none overflow-hidden max-h-60"
      {...inputProps}
    />
  );
};

const Button: FC<ButtonProps> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 disabled:bg-gray-300 bg-green hover:bg-green-light text-black rounded-xl py-2.5 px-8 transition-colors font-semibold"
    {...buttonProps}
  />
);

const ResultsContainer = createDivContainer(`mt-16 w-full max-w-[512px]`);

const DownloadButtons = createDivContainer(``);

const Footer: FC = () => {
  return (
    <footer
      className="w-full bg-green border-t-4 border-black/75 hover:bg-green-light text-black text-xl py-2.5 px-8 fixed bottom-0  cursor-pointer select-none z-50"
      onClick={() => (window.location.href = "https://www.modal.com")}
    >
      Powered by Modal
    </footer>
  );
};
