"use client";

import { cn } from "@/lib/utils";
import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload } from "lucide-react";
import { useDropzone } from "react-dropzone";

const mainVariant = {
  initial: {
    x: 0,
    y: 0,
  },
  animate: {
    x: 20,
    y: -20,
    opacity: 0.9,
  },
};

const secondaryVariant = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
  },
};

export const FileUpload = ({
  onChange,
  accept = {
    "application/pdf": [".pdf"],
  },
  maxFiles = 1,
}: {
  onChange?: (files: File[]) => void;
  accept?: Record<string, string[]>;
  maxFiles?: number;
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (newFiles: File[]) => {
    // Replace existing files if maxFiles is 1, otherwise append
    if (maxFiles === 1) {
      setFiles(newFiles.slice(0, 1));
      onChange && onChange(newFiles.slice(0, 1));
    } else {
      const combinedFiles = [...files, ...newFiles].slice(0, maxFiles);
      setFiles(combinedFiles);
      onChange && onChange(newFiles);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const { getRootProps, isDragActive } = useDropzone({
    multiple: maxFiles > 1,
    noClick: true,
    onDrop: handleFileChange,
    accept,
    maxFiles,
    onDropRejected: (error) => {
      console.log(error);
    },
  });

  // Create accept string for the input element from the accept object
  const getAcceptString = () => {
    return Object.entries(accept)
      .flatMap(([mimeType, extensions]) => [mimeType, ...extensions])
      .join(",");
  };

  return (
    <div className="w-full" {...getRootProps()}>
      <motion.div
        onClick={handleClick}
        whileHover="animate"
        className="p-10 group/file block rounded-lg cursor-pointer w-full relative overflow-hidden"
      >
        <input
          ref={fileInputRef}
          id="file-upload-handle"
          type="file"
          accept={getAcceptString()}
          multiple={maxFiles > 1}
          onChange={(e) => handleFileChange(Array.from(e.target.files || []))}
          className="hidden"
        />
        <div className="absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,white,transparent)]">
          <GridPattern />
        </div>
        <div className="flex flex-col items-center justify-center">
          <p className="relative z-20 font-sans font-bold text-neutral-700 dark:text-neutral-300 text-base">
            Upload PDF
          </p>
          <p className="relative z-20 font-sans font-normal text-neutral-400 dark:text-neutral-400 text-base mt-2">
            {maxFiles === 1
              ? "Drag or drop your PDF file here or click to upload"
              : `Drag or drop up to ${maxFiles} PDF files here or click to upload`}
          </p>
          <div className="relative w-full mt-10 max-w-xl mx-auto">
            {files.length > 0 &&
              files.map((file, idx) => (
                <motion.div
                  key={"file" + idx}
                  layoutId={idx === 0 ? "file-upload" : "file-upload-" + idx}
                  className={cn(
                    "relative overflow-hidden z-40 bg-gray-200 dark:bg-neutral-800 flex flex-col items-start justify-start md:h-24 p-4 mt-4 w-full mx-auto rounded-md", // Darker background in light mode
                    "shadow-sm border border-gray-300 dark:border-neutral-700" // Added border for better definition
                  )}
                >
                  <div className="flex justify-between w-full items-center gap-4">
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      layout
                      className="text-base font-medium text-neutral-800 dark:text-neutral-100 truncate max-w-xs" // Darker text in light mode
                    >
                      {file.name}
                    </motion.p>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      layout
                      className="rounded-lg px-2 py-1 w-fit shrink-0 text-sm bg-gray-300 text-neutral-800 dark:bg-neutral-700 dark:text-neutral-100 shadow-input" // Added bg color for light mode
                    >
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </motion.p>
                  </div>

                  <div className="flex text-sm md:flex-row flex-col items-start md:items-center w-full mt-2 justify-between text-neutral-700 dark:text-neutral-400">
                    {" "}
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      layout
                      className="px-1 py-0.5 rounded-md bg-gray-300 text-neutral-800 dark:bg-neutral-700 dark:text-neutral-300" // Enhanced contrast
                    >
                      {file.type}
                    </motion.p>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      layout
                      className="font-medium" 
                    >
                      modified{" "}
                      {new Date(file.lastModified).toLocaleDateString()}
                    </motion.p>
                  </div>
                </motion.div>
              ))}
            {!files.length && (
              <motion.div
                layoutId="file-upload"
                variants={mainVariant}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 20,
                }}
                className={cn(
                  "relative group-hover/file:shadow-2xl z-40 bg-gray-100 dark:bg-neutral-900 flex items-center justify-center h-32 mt-4 w-full max-w-[8rem] mx-auto rounded-md", // Changed from white to gray-100
                  "shadow-[0px_10px_50px_rgba(0,0,0,0.1)] border border-gray-300 dark:border-neutral-800" // Added border
                )}
              >
                {isDragActive ? (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-neutral-700 dark:text-neutral-300 flex flex-col items-center" // Darker text in light mode
                  >
                    Drop it
                    <Upload className="h-4 w-4 text-neutral-700 dark:text-neutral-300" />
                  </motion.p>
                ) : (
                  <Upload className="h-4 w-4 text-neutral-700 dark:text-neutral-300" />
                )}
              </motion.div>
            )}

            {!files.length && (
              <motion.div
                variants={secondaryVariant}
                className="absolute opacity-0 border border-dashed border-sky-400 inset-0 z-30 bg-transparent flex items-center justify-center h-32 mt-4 w-full max-w-[8rem] mx-auto rounded-md"
              ></motion.div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export function GridPattern() {
  const columns = 41;
  const rows = 11;
  return (
    <div className="flex bg-gray-100 dark:bg-neutral-900 shrink-0 flex-wrap justify-center items-center gap-x-px gap-y-px scale-105">
      {Array.from({ length: rows }).map((_, row) =>
        Array.from({ length: columns }).map((_, col) => {
          const index = row * columns + col;
          return (
            <div
              key={`${col}-${row}`}
              className={`w-10 h-10 flex shrink-0 rounded-[2px] ${
                index % 2 === 0
                  ? "bg-gray-50 dark:bg-neutral-950"
                  : "bg-gray-50 dark:bg-neutral-950 shadow-[0px_0px_1px_3px_rgba(255,255,255,1)_inset] dark:shadow-[0px_0px_1px_3px_rgba(0,0,0,1)_inset]"
              }`}
            />
          );
        })
      )}
    </div>
  );
}
