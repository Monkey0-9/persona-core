import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBoundaryProps {
  title?: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    errorMessage: '',
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error.message || 'Unexpected rendering error.',
    };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo): void {
    // Error details intentionally suppressed from DOM stack exposure
  }

  handleReset = (): void => {
    this.setState({ hasError: false, errorMessage: '' });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-[#BE123C]">
            <AlertTriangle className="w-4 h-4" />
            <span>{this.props.title ?? 'Panel Render Error'}</span>
          </div>
          <p className="text-xs text-[#5C5B56]">{this.state.errorMessage}</p>
          <button
            type="button"
            onClick={this.handleReset}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#141413] text-white rounded-lg hover:bg-[#2B2B29] cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Recover View</span>
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
