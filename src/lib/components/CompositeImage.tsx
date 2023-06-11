import { DetailedHTMLProps, ImgHTMLAttributes, memo, useState } from "react";
import { FC } from "../types";

type CompositeImageProps = {
  imgSrc: string;
  qrCodeDataURL: string;
};

const CompositeImageComp: FC<CompositeImageProps> = ({
  imgSrc,
  qrCodeDataURL,
}) => {
  const [sliderValue, setSliderValue] = useState(50);
  return (
    <Container>
      <Image src={qrCodeDataURL} />
      <Slider value={sliderValue} onChange={setSliderValue} />
    </Container>
  );
};

export const CompositeImage = memo(CompositeImageComp);

const Container: FC = ({ children }) => (
  <div className="w-full max-w-lg">{children}</div>
);

type SliderProps = {
  value: number;
  onChange: (newVal: number) => void;
};

const Slider: FC<SliderProps> = ({ value, onChange }) => (
  <input
    type="range"
    className="w-full"
    value={value}
    onChange={(e) => onChange(Number(e.target.value))}
  />
);

const Image: FC<
  DetailedHTMLProps<ImgHTMLAttributes<HTMLImageElement>, HTMLImageElement>
> = ({ ...imgProps }) => <img className="w-full" {...imgProps} />;
