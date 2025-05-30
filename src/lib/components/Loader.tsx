import _ from "lodash";
import { memo, useEffect, useRef, useState } from "react";
import { FC } from "../types";
import { LOADING_MESSAGES, LOADING_MESSAGE_INTERVAL_MS } from "../../config";
import { createDivContainer } from "./helpers";

const LoaderComp: FC = () => {
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const loadingMessageRef = useRef<string | null>(null);

  useEffect(() => {
    displayLoadingMessage();
    const intervalID = setInterval(
      displayLoadingMessage,
      LOADING_MESSAGE_INTERVAL_MS
    );

    return () => {
      clearInterval(intervalID);
    };

    function displayLoadingMessage(): void {
      if (LOADING_MESSAGES.length <= 1) {
        const m = LOADING_MESSAGES[0] ?? `Loading...`;
        loadingMessageRef.current = m;
        setLoadingMessage(m);
        return;
      }

      // TODO: this is so dumb
      while (true) {
        const i = _.random(0, LOADING_MESSAGES.length - 1);
        const m = LOADING_MESSAGES[i];

        if (m !== loadingMessageRef.current) {
          loadingMessageRef.current = m;
          setLoadingMessage(m);
          return;
        }
      }
    }
  }, []);

  return (
    <Container>
      {loadingMessage && <LoadingMessage>{loadingMessage}</LoadingMessage>}
    </Container>
  );
};

export const Loader = memo(LoaderComp);

const Container = createDivContainer(
  "flex items-center justify-center w-full max-w-lg h-96"
);

const LoadingMessage = createDivContainer("text-sm font-pixel text-green-font");
