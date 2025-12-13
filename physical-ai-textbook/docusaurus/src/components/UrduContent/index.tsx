import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function UrduContent({ content }) {
  return (
    <div dir="rtl">
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({children, ...props}) => <code dir="ltr" {...props}>{children}</code>,
          pre: ({children, ...props}) => <pre dir="ltr" {...props}>{children}</pre>
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
