import {
  ButtonHTMLAttributes,
  DetailedHTMLProps,
  FormHTMLAttributes,
  HTMLAttributes,
  ImgHTMLAttributes,
  InputHTMLAttributes,
  PropsWithChildren,
  FC as ReactFC,
  forwardRef,
} from "react";

export type FC<Props = object> = ReactFC<PropsWithChildren<Props>>;

export type FRC<Element, Props = object> = ReturnType<
  typeof forwardRef<Element, PropsWithChildren<Props>>
>;

export type DivProps = DetailedHTMLProps<
  HTMLAttributes<HTMLDivElement>,
  HTMLDivElement
>;

export type ImgProps = DetailedHTMLProps<
  ImgHTMLAttributes<HTMLImageElement>,
  HTMLImageElement
>;

export type FormProps = DetailedHTMLProps<
  FormHTMLAttributes<HTMLFormElement>,
  HTMLFormElement
>;

export type InputProps = DetailedHTMLProps<
  InputHTMLAttributes<HTMLInputElement>,
  HTMLInputElement
>;

export type ButtonProps = DetailedHTMLProps<
  ButtonHTMLAttributes<HTMLButtonElement>,
  HTMLButtonElement
>;
