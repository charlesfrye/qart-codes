import { forwardRef, memo, useLayoutEffect, useRef, useState } from "react";
import { DivProps, FC, FRC, ImgProps } from "../types";

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
      <Composite
        imgSrcLeft={qrCodeDataURL}
        imgSrcRight={imgSrc}
        position={sliderValue}
      />
      <Slider value={sliderValue} onChange={setSliderValue} />
    </Container>
  );
};

export const CompositeImage = memo(CompositeImageComp);

type CompositeProps = {
  imgSrcLeft: string;
  imgSrcRight: string;
  position: number;
};

const CompositeComp: FC<CompositeProps> = ({
  imgSrcLeft,
  imgSrcRight,
  position,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number | null>(null);

  useLayoutEffect(() => {
    updateContainerWidth();
    window.addEventListener(`resize`, updateContainerWidth);
    return () => {
      window.removeEventListener(`resize`, updateContainerWidth);
    };

    function updateContainerWidth(): void {
      const containerEl = containerRef.current;
      if (containerEl == null) {
        return;
      }
      setContainerWidth(containerEl.clientWidth);
    }
  }, []);

  return (
    <CompositeContainer
      ref={containerRef}
      style={{ height: containerWidth ?? 0 }}
    >
      {containerWidth != null && (
        <>
          <ImageContainer widthPercentage={position}>
            <Image style={{ width: containerWidth }} src={imgSrcLeft} />
          </ImageContainer>
          <ImageContainer widthPercentage={100 - position}>
            <Image
              style={{
                width: containerWidth,
                right: 0,
              }}
              src={imgSrcRight}
            />
          </ImageContainer>
        </>
      )}
    </CompositeContainer>
  );
};

const Composite = memo(CompositeComp);

const Container: FC = ({ children }) => (
  <div className="w-full max-w-lg">{children}</div>
);

const CompositeContainer: FRC<HTMLDivElement, DivProps> = forwardRef(
  ({ children, ...divProps }, ref) => (
    <div className="flex justify-between" ref={ref} {...divProps}>
      {children}
    </div>
  )
);

type ImageContainerProps = {
  widthPercentage: number;
};

const ImageContainer: FC<ImageContainerProps> = ({
  widthPercentage,
  children,
}) => (
  <div
    className={`overflow-hidden shrink-0 relative`}
    style={{ width: `${widthPercentage}%` }}
  >
    {children}
  </div>
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

const Image: FC<ImgProps> = ({ ...imgProps }) => (
  <img className="max-w-none aspect-square absolute" {...imgProps} />
);
