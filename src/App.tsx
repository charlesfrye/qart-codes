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
<div className="w-full max-w-3xl mx-auto px-5 mb-4">
  <div className="flex items-center justify-between gap-2 w-full text-sm sm:text-base">
    <div className="flex items-center gap-1 sm:gap-2 text-green-light font-degular whitespace-nowrap overflow-hidden">
      <span className="truncate">Built with</span>
      <img
        src="/modal_clear_logo.svg"
        alt="Modal Logo"
        className="h-4 w-auto sm:h-5"
      />
      <span className="truncate">Modal</span>
    </div>
    <a
      href="https://modal.com/playground"
      target="_blank"
      rel="noopener noreferrer"
      className="min-w-0 shrink max-w-[50%]"
    >
      <button
        className="
          flex items-center gap-1 sm:gap-2
          bg-green-bright rounded-lg
          px-2 sm:px-3 py-1
          text-xs sm:text-14 font-inter
          whitespace-nowrap w-full overflow-hidden
        "
      >
        <span className="truncate">Get Started</span>
        <img
          src="top-right_arrow.svg"
          className="h-4 w-auto shrink-0"
        />
      </button>
    </a>
  </div>
</div>
		<div className="w-full max-w-3xl mx-auto bg-gray border-[0.5px] border-[rgba(127,238,100,0.2)] rounded-lg p-8">
				<div className="flex items-start justify-between mb-4">
					<div>
						<img
      src="/q-art_logo.svg"
      alt="Q-Art Codes Logo"
      className="w-40 md:w-56 lg:w-64 h-auto drop-shadow-xl"
    />
    <div className="mt-2 text-xs font-inter font-style: italic text-green-light">
      Create QR Codes with aesthetically pleasing corruptions
    </div>
  </div>
	<a
  href="https://github.com/charlesfrye/qart-codes"
  target="_blank"
  rel="noopener noreferrer"
  className="inline-flex items-center justify-center gap-2 text-green-light font-degular border border-green-light rounded-full px-3 py-1.5 leading-none whitespace-nowrap max-w-full text-sm sm:text-base"
>
  <img
    src="/GitHubIcon.svg"
    alt="GitHub Icon"
    className="w-4 h-4 shrink-0"
  />
  <span className="relative top-[0.5px]">View Code</span>
</a>
				</div>
				<UserInput>
  <div> 
    <div className="flex flex-col gap-2">
      <label
        htmlFor="prompt"
        className="text-3xl font-degular font-light text-green-light"
      >
        Prompt
      </label>
      <Textarea
        placeholder="Visual content or style to apply to the QR code"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
    </div>

    <div className="flex flex-col gap-2 mt-6">
      <label
        htmlFor="qrValue"
        className="text-3xl font-degular font-light text-green-light"
      >
        Link
      </label>
      <Input
        id="qrValue"
        placeholder="Text to encode, like a URL or your Wi-Fi password"
        value={qrCodeValue}
        onChange={(e) => setQRCodeValue(e.target.value)}
      />
    </div>

    <Button disabled={loading} onClick={loading ? cancel : generate}>
      {loading ? "Cancel Generation" : "Generate Q-Art Code"}
    </Button>
  </div>
</UserInput>
      {(loading || (imgSrc && qrCodeDataURL)) && (
        <ResultsContainer>
          {loading && <Loader />}
          {imgSrc && qrCodeDataURL && (
            <>
						<div className="flex flex-col md:flex-row justify-center items-start gap-6 w-full">
							<div className="flex-shrink-0 max-w-[512px] w-full">
              <CompositeImage imgSrc={imgSrc} qrCodeDataURL={qrCodeDataURL} />
							</div>
							<DownloadButtons>
                <SmallButton onClick={downloadQArtCode}>
									<img src='/download_icon.svg' className="w-4 h-4 opacity-[0.40]"/>
									Download Q-Art Code
									</SmallButton>
                <SmallButton onClick={downloadQRCode}>
									<img src='/download_icon.svg' className="w-4 h-4 opacity-[0.40]"/>
									Download QR Code
									</SmallButton>
              </DownloadButtons>
							</div>
            </>
          )}
        </ResultsContainer>
      )}
			</div>
    </Container>
  );
}

export default App;

const Container = createDivContainer(
	"min-h-screen w-full flex flex-col items-center justify-center p-4"
);

const UserInput: FC<FormProps> = ({ children }) => (
  <form onSubmit={(e) => e.preventDefault()}>{children}</form>
);

const Input: FC<InputProps> = ({ ...inputProps }) => (
  <input
    className="w-full rounded-xl py-2.5 px-8 first:mt-0
               bg-green-light/10       
               text-green-light font-degular 
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
      className="w-full rounded-xl py-2.5 px-8 first:mt-0
                 bg-green-light/10      
                 text-green-light font-degular font-light leading-relaxed
                 placeholder:text-green-light/60
                 focus:outline-none resize-none overflow-hidden max-h-60"
      {...inputProps}
    />
  );
};

const Button: FC<ButtonProps> = ({ ...buttonProps }) => (
  <button
    className="w-full mt-4 disabled:bg-gray-300 bg-[#7FEE64] … rounded-xl py-2.5 px-8 transition-colors"
    {...buttonProps}
  />
);

const SmallButton: FC<ButtonProps> = ({ className = "", ...buttonProps }) => (
  <button
    className={`
			flex items-center gap-1
      bg-green-light/5 text-green-light/40 text-sm whitespace-nowrap
      rounded-md px-6 py-2 transition-colors 
			border border-green-light/5 ${className}
    `}
    {...buttonProps}
  />
);


const ResultsContainer = createDivContainer(`mt-10 w-full max-w-3xl`);

const DownloadButtons = createDivContainer(`flex flex-col gap-2 mt-2`);


