"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { BackgroundLines } from "@/components/ui/background-lines";
import ColourfulText from "@/components/ui/colourful-text";
import { Navbar } from "@/components/Navbar";
import { Features } from "./Features";
import { ContainerScroll } from "@/components/ui/container-scroll-animation";

export default function HomePage() {
  const router = useRouter();

  return (
    <>
      <BackgroundLines>
        <div className="relative mx-auto my-10 flex max-w-7xl flex-col items-center justify-center">
          <Navbar />
          <div className="px-4 py-10 md:py-20 text-center">
            <h1 className="relative z-10 mx-auto max-w-4xl text-2xl font-bold text-slate-700 md:text-4xl lg:text-7xl dark:text-slate-300">
              {"AI-Powered Legal Document".split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.3,
                    delay: index * 0.1,
                    ease: "easeInOut",
                  }}
                  className="mr-2 inline-block"
                >
                  {word}
                </motion.span>
              ))}
              {"Risk Analysis".split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.3,
                    delay: index * 0.1,
                    ease: "easeInOut",
                  }}
                  className="mr-2 inline-block"
                >
                  <ColourfulText text={word} />
                </motion.span>
              ))}
            </h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.8 }}
              className="mt-4 max-w-2xl mx-auto text-lg text-neutral-600 dark:text-neutral-400"
            >
              Upload your legal documents and let our advanced AI analyze them
              for risks, helping you make informed decisions with confidence.
            </motion.p>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 1 }}
              className="mt-8 flex flex-wrap justify-center gap-4"
            >
              <button
                onClick={() => router.push("/analysis")}
                className="w-60 rounded-lg bg-blue-600 px-6 py-3 text-white font-medium transition-all duration-300 hover:bg-blue-700"
              >
                Get Started
              </button>
              <button
                onClick={() => router.push("/about")}
                className="w-60 rounded-lg border border-gray-300 bg-white px-6 py-3 font-medium text-black transition-all duration-300 hover:bg-gray-100 dark:border-gray-700 dark:bg-black dark:text-white dark:hover:bg-gray-900"
              >
                Learn More
              </button>
            </motion.div>
            <ContainerScroll
              titleComponent={
                <>
                  <h1 className="text-4xl font-semibold text-black dark:text-white">
                    Unleash the power of <br />
                    <span className="text-4xl md:text-[6rem] font-bold mt-1 leading-none">
                      <ColourfulText text="Deep Learning" />
                    </span>
                  </h1>
                </>
              }
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.2 }}
                className="mt-16 w-full max-w-4xl mx-auto border rounded-lg shadow-lg overflow-hidden"
              >
                <Image
                  src="/vercel.svg"
                  alt="Legal Document Analysis Preview"
                  className="w-full h-auto"
                  width={1000}
                  height={500}
                />
              </motion.div>
            </ContainerScroll>

            <Features />
          </div>
        </div>
      </BackgroundLines>
    </>
  );
}
