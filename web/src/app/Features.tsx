import { HoverEffect } from "@/components/ui/card-hover-effect";

import {
  FileText,
  Upload,
  Brain,
  ShieldCheck,
  FileCheck,
  ScanLine,
} from "lucide-react";

const features: {
  icon: React.ReactNode;
  title: string;
  description: string;
}[] = [
  {
    icon: <Upload size={24} />,
    title: "PDF Upload & Storage",
    description: "Securely upload legal documents to AWS S3 for analysis.",
  },
  {
    icon: <ScanLine size={24} />,
    title: "AI-Powered Risk Detection",
    description: "Detects risky clauses and compliance issues in contracts.",
  },
  {
    icon: <Brain size={24} />,
    title: "Fine-Tuned BERT Model",
    description: "Uses NLP to classify legal clauses based on risk levels.",
  },
  {
    icon: <FileText size={24} />,
    title: "Clause-Based Analysis",
    description: "Breaks documents into sections for precise risk assessment.",
  },
  {
    icon: <FileCheck size={24} />,
    title: "Comprehensive Reports",
    description: "Generates detailed risk reports for legal review.",
  },
  {
    icon: <ShieldCheck size={24} />,
    title: "Secure & Scalable",
    description: "Built on AWS & Next.js for security and performance.",
  },
];

export const Features = () => {
  return (
    <div className="max-w-5xl mx-auto px-8">
      <HoverEffect items={features} />
    </div>
  );
};
