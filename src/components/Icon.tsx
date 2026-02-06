import React from 'react';
import * as LucideIcons from 'lucide-react';
import type {LucideProps} from 'lucide-react';

type LucideComponent = React.ComponentType<LucideProps>;
type IconRegistry = Record<string, LucideComponent>;

export interface IconProps extends LucideProps {
  name: string;
  fallbackName?: string;
}

const icons = LucideIcons as unknown as IconRegistry;

export default function Icon({
  name,
  fallbackName = 'CircleHelp',
  size = 18,
  strokeWidth = 2,
  ...rest
}: IconProps): React.JSX.Element {
  const SelectedIcon = icons[name] ?? icons[fallbackName] ?? LucideIcons.CircleHelp;
  return <SelectedIcon size={size} strokeWidth={strokeWidth} aria-hidden='true' {...rest} />;
}