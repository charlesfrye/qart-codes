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

function App() {
  const [prompt, setPrompt] = useState(``);
  const [qrCodeValue, setQRCodeValue] = useState(``);
  const [qrCodeDataURL, setQRCodeDataURL] = useState<string | null>(null);

  const generate = useCallback(async () => {
    const dataURL = await generateQRCodeDataURL();
    setQRCodeDataURL(dataURL);
  }, []);

  return (
    <Container>
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
      {qrCodeDataURL && (
        <CompositeImage
          imgSrc={PLACEHOLDER_IMAGE_URL}
          qrCodeDataURL={qrCodeDataURL}
        />
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

const Input: FC<
  DetailedHTMLProps<InputHTMLAttributes<HTMLInputElement>, HTMLInputElement>
> = ({ ...inputProps }) => <input className="w-full" {...inputProps} />;

const Button: FC<
  DetailedHTMLProps<ButtonHTMLAttributes<HTMLButtonElement>, HTMLButtonElement>
> = ({ ...buttonProps }) => <button className="w-full" {...buttonProps} />;

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
