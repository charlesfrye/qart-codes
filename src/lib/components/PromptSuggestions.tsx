import { FC } from "react";

interface PromptChipProps {
  text: string;
  onClick: (text: string) => void;
}

const PromptChip: FC<PromptChipProps> = ({ text, onClick }) => (
    <button
      onClick={() => onClick(text)}
      className="
        px-4 py-2 rounded-full
        bg-green-light/10 text-green-light/70
        text-xs font-inter
        hover:bg-green-light/20 hover:text-green-light
        transition-colors border border-green-light/20
        max-w-[200px] truncate
        flex-shrink-0
      "
      title={text}
    >
      {text}
    </button>
  );

interface PromptSuggestionsProps {
  prompts: string[];
  onPromptSelect: (prompt: string) => void;
}

export const PromptSuggestions: FC<PromptSuggestionsProps> = ({
    prompts,
    onPromptSelect
  }) => (
    <div className="flex flex-wrap gap-2 mt-3">
      {prompts.map((prompt, idx) => (
        <PromptChip
          key={idx}
          text={prompt}
          onClick={onPromptSelect}
        />
      ))}
    </div>
  );
