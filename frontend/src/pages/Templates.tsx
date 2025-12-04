import Layout from '../components/Layout';
import Card from '../components/Card';
import { useNavigate } from 'react-router-dom';

const templates = [
  {
    title: 'Exam Revision Notes',
    desc: 'Template to capture formulae, definitions and one-line summaries topic wise.'
  },
  {
    title: 'Chapter Summary',
    desc: 'Summarise any chapter into key ideas, characters, events and quotes.'
  },
  {
    title: 'Question Bank',
    desc: 'Prepare likely exam questions and important 2/5/10 marks questions.'
  },
  {
    title: 'Daily Study Log',
    desc: 'Track what you studied each day and what needs revision.'
  }
];

export default function Templates() {
  const navigate = useNavigate();
  const handleUseTemplate = (template: any) => navigate('/tools', { state: { template } });

  return (
    <Layout>
      <div className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Templates</div>
        <div className="text-xl font-semibold mt-1">Premium study templates</div>
        <p className="text-sm text-gray-400 max-w-xl mt-1">
          Pick a template as a starting point and then fill it with content from your subjects and chapters.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {templates.map((t) => (
          <Card key={t.title}>
            <div className="flex flex-col gap-3">
              <div className="text-sm font-semibold">{t.title}</div>
              <p className="text-xs text-gray-400">{t.desc}</p>
              <button onClick={() => handleUseTemplate(t)} className="mt-1 text-[11px] px-3 py-1.5 rounded-full bg-violet-500/90 hover:bg-violet-400 text-white w-fit">
                Use this template
              </button>
            </div>
          </Card>
        ))}
      </div>
    </Layout>
  );
}
