import {
  forwardRef,
  memo,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { DivProps, FC, FRC, ImgProps } from "../types";
import { createDivContainer } from "./helpers";
import { IconDownload } from "./svg/IconDownload";
import { downloadStitchedImage } from "../download";

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

  const downloadStitched = useCallback(async () => {
    await downloadStitchedImage(imgSrcLeft, imgSrcRight, position / 100);
  }, [imgSrcLeft, imgSrcRight, position]);

  return (
    <CompositeContainer
      ref={containerRef}
      style={{ height: containerWidth ?? 0 }}
    >
      {containerWidth != null && (
        <>
          <ImageContainer style={{ width: `${position}%` }}>
            <Image style={{ width: containerWidth }} src={imgSrcLeft} />
          </ImageContainer>
          <ImageContainer style={{ width: `${100 - position}%` }}>
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
      <DownloadOverlay onClick={downloadStitched}>
        <IconDownloadContainer>
          <IconDownload />
        </IconDownloadContainer>
      </DownloadOverlay>
    </CompositeContainer>
  );
};

const Composite = memo(CompositeComp);

const Container = createDivContainer("w-full max-w-[512px] flex flex-col");

const CompositeContainer: FRC<HTMLDivElement, DivProps> = forwardRef(
  ({ children, ...divProps }, ref) => (
    <div className="relative flex justify-between" ref={ref} {...divProps}>
      {children}
    </div>
  )
);

const DownloadOverlay = createDivContainer(
  `absolute inset-0 bg-black bg-opacity-25 z-10 cursor-pointer transition-opacity opacity-0 hover:opacity-100`
);

const IconDownloadContainer = createDivContainer(
  `w-8 h-8 top-4 right-4 absolute text-white`
);

const ImageContainer = createDivContainer(`overflow-hidden shrink-0 relative`);

type SliderProps = {
  value: number;
  onChange: (newVal: number) => void;
};

const Slider: FC<SliderProps> = ({ value, onChange }) => (
  <input
    type="range"
    className="w-full mt-4"
    value={value}
    onChange={(e) => onChange(Number(e.target.value))}
  />
);

const Image: FC<ImgProps> = ({ ...imgProps }) => (
  <img className="max-w-none aspect-square absolute" {...imgProps} />
);
