import { DivProps, FC } from "../types";

export function createDivContainer(className: string): FC<DivProps> {
  return ({ children, ...props }) => (
    <div className={className} {...props}>
      {children}
    </div>
  );
}
