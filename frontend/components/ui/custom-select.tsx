"use client";

import * as Select from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { useId, useMemo } from "react";

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

const EMPTY_VALUE = "__open_research_empty_select_value__";

export function CustomSelect({
  value,
  onChange,
  options,
  placeholder = "Select an option",
  ariaLabel,
  disabled = false,
}: CustomSelectProps) {
  const listboxId = useId();
  const triggerLabelId = `${listboxId}-label`;
  const selectValue = value === "" ? EMPTY_VALUE : value;

  const selectedOption = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );

  return (
    <div className={`custom-select ${disabled ? "disabled" : ""}`}>
      <span className="sr-only" id={triggerLabelId}>
        {ariaLabel ?? placeholder}
      </span>
      <Select.Root
        disabled={disabled}
        onValueChange={(nextValue) => onChange(nextValue === EMPTY_VALUE ? "" : nextValue)}
        value={selectValue}
      >
      <Select.Trigger
        aria-controls={listboxId}
        aria-label={ariaLabel}
        aria-labelledby={triggerLabelId}
        className="custom-select-trigger"
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
        <Select.Icon className="custom-select-chevron">
          <ChevronDown aria-hidden size={14} strokeWidth={2} />
        </Select.Icon>
      </Select.Trigger>

      <Select.Portal>
        <Select.Content
          className="custom-select-menu"
          id={listboxId}
          onEscapeKeyDown={(event) => {
            event.stopPropagation();
          }}
          position="popper"
          sideOffset={4}
        >
          <Select.Viewport className="custom-select-viewport">
          {options.map((option) => {
            const isSelected = option.value === value;
            const optionValue = option.value === "" ? EMPTY_VALUE : option.value;
            return (
              <Select.Item
                className={`custom-select-option ${isSelected ? "selected" : ""}`}
                data-selected={isSelected ? "true" : undefined}
                key={option.value}
                value={optionValue}
              >
                <Select.ItemText>
                  <span className="custom-select-option-label">{option.label}</span>
                </Select.ItemText>
                {option.description ? (
                  <span className="custom-select-option-description">
                    {option.description}
                  </span>
                ) : null}
                <Select.ItemIndicator className="custom-select-option-indicator">
                  <Check aria-hidden size={14} strokeWidth={2} />
                </Select.ItemIndicator>
              </Select.Item>
            );
          })}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
      </Select.Root>
    </div>
  );
}
