"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

export interface CustomSelectOption {
  value: string;
  label: string;
  description?: string;
}

interface CustomSelectProps {
  value: string;
  onChange: (value: string) => void;
  options: ReadonlyArray<CustomSelectOption>;
  placeholder?: string;
  ariaLabel?: string;
  disabled?: boolean;
}

export function CustomSelect({
  value,
  onChange,
  options,
  placeholder = "Select an option",
  ariaLabel,
  disabled = false,
}: CustomSelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();

  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [open]);

  return (
    <div className={`custom-select ${open ? "open" : ""} ${disabled ? "disabled" : ""}`} ref={rootRef}>
      <button
        aria-controls={listboxId}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        className="custom-select-trigger"
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <span className="custom-select-trigger-copy">
          <span className="custom-select-value">
            {selectedOption?.label ?? placeholder}
          </span>
          {selectedOption?.description ? (
            <span className="custom-select-value-description">
              {selectedOption.description}
            </span>
          ) : null}
        </span>
        <span aria-hidden className="custom-select-chevron">
          {open ? "−" : "+"}
        </span>
      </button>

      {open ? (
        <div className="custom-select-menu" id={listboxId} role="listbox">
          {options.map((option) => {
            const isSelected = option.value === value;
            return (
              <button
                aria-selected={isSelected}
                className={`custom-select-option ${isSelected ? "selected" : ""}`}
                key={option.value}
                onClick={() => {
                  onChange(option.value);
                  setOpen(false);
                }}
                role="option"
                type="button"
              >
                <span className="custom-select-option-label">{option.label}</span>
                {option.description ? (
                  <span className="custom-select-option-description">
                    {option.description}
                  </span>
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
