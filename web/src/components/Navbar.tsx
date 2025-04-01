import { Upload } from "lucide-react";
import Link from "next/link";

export const Navbar = () => {
  return (
    <nav className="w-full flex items-center justify-between border-b border-neutral-200 px-6 py-4 dark:border-neutral-800">
      <div className="flex items-center gap-2">
        <div className="size-7 rounded-full bg-gradient-to-br from-blue-500 to-indigo-500" />
        <h1 className="text-lg font-bold md:text-2xl">LegalLens</h1>
      </div>
      <Link href={"/analysis"}>
        <button className="flex items-center gap-2 w-36 rounded-lg bg-blue-600 px-4 py-2 font-medium text-white transition-all duration-300 hover:bg-blue-700 dark:bg-white dark:text-black dark:hover:bg-gray-200">
          <Upload className="w-5 h-5 mr-2" />
          Upload
        </button>
      </Link>
    </nav>
  );
};
