import { FC as ReactFC, PropsWithChildren } from "react";

export type FC<Props = object> = ReactFC<PropsWithChildren<Props>>;
