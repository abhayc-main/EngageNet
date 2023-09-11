import "./globals.css";
import { Analytics } from "@vercel/analytics/react";
import cx from "classnames";
import { sfPro, inter } from "./fonts";
import Footer from "@/components/layout/footer";
import { Suspense } from "react";

export const metadata = {
  title: "EngageNet",
  description:
    "AI based crowd engagement tracking",
  themeColor: "#FFF",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={cx(sfPro.variable, inter.variable, "bg-gradient-to-br from-indigo-50 via-white to-cyan-100")}>
        <main>
          {children}
        </main>
        <Footer />
        <Analytics />
      </body>
    </html>
  );
}
