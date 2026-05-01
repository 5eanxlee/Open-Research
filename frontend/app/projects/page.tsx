import { ResearchDashboard } from "@/components/research-dashboard";

interface ProjectsPageProps {
  searchParams: Promise<{
    projectId?: string;
  }>;
}

export default async function ProjectsPage({ searchParams }: ProjectsPageProps) {
  const { projectId = null } = await searchParams;

  return <ResearchDashboard initialProjectId={projectId} />;
}
